# Add after existing globals around line 50
pending_location_requests = {}  # {ticket_id: ticket_data}
awaiting_location_response = {}  # {ticket_id: True/False}



# ---------------------- Automation Control Globals ----------------------
automation_running = False
automation_lock = threading.Lock()
shutdown_event = asyncio.Event()
stop_event = None  # Will be passed from the server
# Add after existing globals around line 50
processing_tickets = set()  # Track tickets currently being processed
processing_lock = threading.Lock()
location_lock = asyncio.Lock()  # Add this line


def set_automation_running(status: bool):
    """Thread-safe way to set automation status"""
    global automation_running
    with automation_lock:
        automation_running = status

def is_automation_running() -> bool:
    """Thread-safe way to check automation status"""
    global automation_running
    with automation_lock:
        return automation_running

def signal_shutdown():
    """Signal the automation to stop"""
    set_automation_running(False)
    shutdown_event.set()
    if stop_event:
        stop_event.set()

async def check_shutdown():
    """Check if shutdown was requested"""
    # Check both internal shutdown and external stop event from server
    internal_stop = shutdown_event.is_set() or not is_automation_running()
    external_stop = stop_event.is_set() if stop_event else False
    
    if external_stop and is_automation_running():
        log("🛑 External stop signal received from server")
        signal_shutdown()
    
    return internal_stop or external_stop


# ---------------------- Logging Setup ----------------------



def handle_location_assistance_response(sr_id: str, response: str, new_location: str = None):
    """Handle response from location assistance request"""
    global pending_location_requests, awaiting_location_response, location_issue_tickets
    
    if sr_id in pending_location_requests:
        if response.lower() == "yes" and new_location:
            ticket_data = pending_location_requests[sr_id]
            ticket_data["location"] = new_location.strip()
            
            # MODIFIED: Add back to processing queue with updated location
            asyncio.create_task(ticket_queue.put(ticket_data))
            log(f"✅ {sr_id} re-queued with new location: {new_location}")
            
            # Clean up
            del pending_location_requests[sr_id]
            if sr_id in awaiting_location_response:
                del awaiting_location_response[sr_id]
            if sr_id in location_issue_tickets:
                location_issue_tickets.remove(sr_id)
                
            send_whatsapp(f"✅ SR {sr_id} re-queued with location: {new_location}")
                
        elif response.lower() == "no":
            # Keep in location issue queue permanently
            log(f"❌ {sr_id} permanently marked as location issue")
            
            # Clean up pending requests
            if sr_id in pending_location_requests:
                del pending_location_requests[sr_id]
            if sr_id in awaiting_location_response:
                del awaiting_location_response[sr_id]
                
            send_whatsapp(f"❌ SR {sr_id} marked as permanent location issue")

# ---------------------- Maximo Setup ----------------------
MAXIMO_BASE = "https://fm.osoolre.com/maximo"
USERNAME    = os.getenv("MAXIMO_USER", "")
PASSWORD    = os.getenv("MAXIMO_PASS", "")
MAXAUTH     = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "OSLC-Core-Version": "2.0",
    "maxauth": MAXAUTH
}
TICKET_FETCH_URL = f"{MAXIMO_BASE}/oslc/os/mxsr"

# ───────── CONFIG ──────────
# Globals
ticket_queue = asyncio.Queue()
wo_queue = asyncio.Queue()
guess_queue  = asyncio.Queue()
location_issue_tickets = set()

NUM_WORKERS  = 3
RETRY_LIMIT  = 8

# Worker tracking
step1_workers_started = False
step2_workers_started = False
step1_guess_started = False
active_step1_workers = 0
active_step2_workers = 0
active1g = 0

# ---------------------- Helper Functions ----------------------

def extract_zone_from_location(location: str) -> str:
    """Extract zone from location - after Z02, skip first '-', get text until second '-', remove letter, use number for zone logic"""
    try:
        # Find "Z02" in the location
        location_upper = location.upper()
        z02_index = location_upper.find('Z02')
        
        if z02_index == -1:
            print(f"⚠️ Z02 not found in location: {location}")
            return "B"  # fallback
        
        # Start after Z02
        remaining = location_upper[z02_index + 3:]
        print(f"🔍 After Z02: '{remaining}'")
        
        # Skip the first '-' if it exists
        if remaining.startswith('-'):
            remaining = remaining[1:]
            print(f"🔍 After skipping first '-': '{remaining}'")
        
        # Find the second '-' and extract everything before it
        second_dash_index = remaining.find('-')
        if second_dash_index == -1:
            print(f"⚠️ No second dash found in: {remaining}")
            return "B"  # fallback
        
        zone_part = remaining[:second_dash_index]
        print(f"🔍 Extracted zone part: '{zone_part}'")
        
        # Remove the letter and extract the number
        # For "B829" we want "829"
        match = re.search(r'(\d+)', zone_part)
        if not match:
            print(f"⚠️ No number found in zone part: {zone_part}")
            return zone_part  # fallback
        
        xi = int(match.group(1))
        print(f"🔍 Extracted number {xi} from zone part: {zone_part}")
        
        # Apply zone logic based on the number
        if 799 <= xi <= 823:
            zone = "A"
        elif 824 <= xi <= 847:
            zone = "B"
        elif 848 <= xi <= 872:
            zone = "C"
        else:
            zone = "B"
            
        print(f"📍 Determined zone: {zone} for number {xi}")
        return zone
        
    except Exception as e:
        print(f"⚠️ Zone extraction error: {e}")
        return "B"  # fallback


# ───── Playwright Automation Functions ────────────────────────────────────────────

async def login_and_open_sr(page, sr_id: str):
    await page.goto(f"{MAXIMO_BASE}/webclient/login/login.jsp")
    await page.fill("#username", USERNAME)
    await page.fill("#password", PASSWORD)
    await page.press("#password", "Enter")
    await page.wait_for_timeout(2000)

    await page.goto(f"{MAXIMO_BASE}/ui/?event=loadapp&value=sr")
    await page.fill("#quicksearch", sr_id)
    await page.press("#quicksearch", "Enter")
    await wait_for_spinner(page)

async def route_sr_and_get_wo(page, sr_id: str, expected_location: str = None):
    await login_and_open_sr(page, sr_id)
    await page.wait_for_selector("#maac272a3-tb", timeout=10000)
    status = await page.input_value("#maac272a3-tb")
    
    if "INPROG" in status:
        await page.click("#m4326cf1d-tab_anchor")
        await wait_for_spinner(page)
        # Wait for the table to be populated
        await page.wait_for_timeout(2000)  # Give it a moment to load
        # More specific selector for the work order input fields
        wo_inputs = page.locator("input[id*='_tdrow_[C:1]_txt-tb[R:']")
        
        # Wait for at least one input to be present
        await wo_inputs.first.wait_for(timeout=10000)
        
        input_count = await wo_inputs.count()
        print(f"Found {input_count} work order inputs")
        
        if input_count == 0:
            raise Exception("No work order inputs found")
        
        # Get the last work order (most recent)
        last_wo_input = wo_inputs.nth(input_count - 1)
        
        # Double-check it's an input element
        is_input = await last_wo_input.evaluate("el => el.tagName.toLowerCase() === 'input'")
        if not is_input:
            raise Exception("Selected element is not an input")
        
        wo = await last_wo_input.input_value()
        
        if not wo or not wo.strip():
            # Try getting the value attribute as fallback
            wo = await last_wo_input.get_attribute("value")
        
        if wo and wo.strip():
            return wo.strip()
        else:
            raise Exception("Work order input is empty")
    
    # NEW: Check and update location if provided and different
    if expected_location:
        try:
            current_location = await page.input_value("#m3dc2199c-tb")
            if current_location.strip() != expected_location.strip():
                log(f"🔄 {sr_id} updating location: {current_location} → {expected_location}")
                await page.fill("#m3dc2199c-tb", expected_location)
                await page.click("#m3dc2199c-img")
                await wait_for_spinner(page)
                await page.click("#m3dc2199c-lb")
                await wait_for_spinner(page)
        except Exception as e:
            log(f"⚠️ {sr_id} location update failed: {e}")
    
    # Route workflow
    await page.click("#ROUTEWF_MAIN-SR_-tbb_anchor")
    await wait_for_spinner(page)
    # Click confirm
    await page.wait_for_selector("#inputwf-dialog_inner", timeout=10000)
    await page.click("#m37917b04-pb")
    # Extract new WO id
    await page.wait_for_selector("#mad3161b5-tb", timeout=10000)
    wo_id = await page.input_value("#mad3161b5-tb")
    return wo_id
async def route_sr_and_get_wo_with_guess(page, sr_id: str, zzlocation: str) -> str:
    """
    Safer version with improved error detection and fallback handling.
    """
    try:
        await login_and_open_sr(page, sr_id)
        
        # Phase 1: Try primary location guess
        success = await _try_location_guess(page, sr_id, zzlocation, is_primary=True)
        
        if not success:
            # Phase 2: Try zone C fallback
            log(f"⚠️ [GUESS] Zone error detected for {sr_id}, trying zone C fallback")
            success = await _try_location_guess(page, sr_id, zzlocation, is_primary=False)
            
            if not success:
                # Phase 3: Both failed - handle assistance request
                await _handle_both_guesses_failed(sr_id, zzlocation)
                raise Exception(f"Both zone guesses failed for {sr_id}, assistance requested")
        
        # Continue with rest of workflow (your existing logic)
        await page.click("#m3dc2199c-lb")
        await wait_for_spinner(page)
        
        await page.click("#ROUTEWF_MAIN-SR_-tbb_anchor")
        await wait_for_spinner(page)
        
        await page.wait_for_selector("#inputwf-dialog_inner", timeout=10000)
        await page.click("#m37917b04-pb")
        
        await page.wait_for_selector("#mad3161b5-tb", timeout=10000)
        wo_id = await page.input_value("#mad3161b5-tb")
        
        if not wo_id or wo_id.strip() == "":
            raise Exception(f"WO ID is empty for {sr_id}")
            
        return wo_id.strip()
        
    except Exception as e:
        log(f"❌ [CRITICAL] Failed to process SR {sr_id}: {str(e)}")
        raise


async def _try_location_guess(page, sr_id: str, zzlocation: str, is_primary: bool = True) -> bool:
    """
    Try a location guess and return True if successful, False if error detected.
    """
    try:
        # Generate the guess
        if is_primary:
            guessed = guess_location_from_zz(zzlocation)
            log(f"🔧 [GUESS] {sr_id} → {guessed}")
        else:
            guessed = guess_location_from_zz(zzlocation, force_zone="C")
            log(f"🔧 [GUESS-RETRY] {sr_id} → {guessed}")
        
        # Clear field and fill with guess
        await _safe_fill_location(page, guessed)
        
        # Click lookup button
        await page.click("#m3dc2199c-img")
        await wait_for_spinner(page)
        await page.click("#m3dc2199c-lb")
        
        # Check for errors using multiple methods
        error_detected = await _check_for_location_errors(page, sr_id)
        
        if error_detected:
            attempt_type = "primary" if is_primary else "fallback"
            log(f"⚠️ [ERROR] {attempt_type} location guess failed for {sr_id}")
            return False
        
        log(f"✅ [SUCCESS] Location guess accepted for {sr_id}")
        return True
        
    except Exception as e:
        attempt_type = "primary" if is_primary else "fallback"
        log(f"⚠️ [EXCEPTION] {attempt_type} guess failed for {sr_id}: {str(e)}")
        return False


async def _safe_fill_location(page, location: str) -> None:
    """Safely fill the location field with retries."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Wait for field to be available
            await page.wait_for_selector("#m3dc2199c-tb", timeout=5000)
            
            # Clear and fill
            await page.fill("#m3dc2199c-tb", location)
            
            # Verify the value was set
            current_value = await page.input_value("#m3dc2199c-tb")
            if current_value == location:
                return  # Success
            
            log(f"⚠️ [FILL] Value mismatch, expected: {location}, got: {current_value}")
            
        except Exception as e:
            if attempt == max_attempts - 1:
                raise Exception(f"Failed to fill location after {max_attempts} attempts: {str(e)}")
            
            log(f"⚠️ [RETRY] Fill attempt {attempt + 1} failed, retrying...")
            await page.wait_for_timeout(1000)


async def _check_for_location_errors(page, sr_id: str) -> bool:
    """
    Comprehensive error checking using multiple detection methods.
    Returns True if error detected, False otherwise.
    """
    try:
        # Method 1: Check the specific error icon visibility (your main case)
        error_icon = await page.query_selector("#mbf28cd64-tab_er_img")
        if error_icon:
            is_visible = await error_icon.is_visible()
            if is_visible:
                log(f"🔍 [ERROR-DETECTED] Error icon visible for {sr_id}")
                return True
        
        # Method 2: Check for system dialog (secondary check)
        system_dialog = await page.query_selector("#msgbox-dialog_inner")
        if system_dialog:
            is_visible = await system_dialog.is_visible()
            if is_visible:
                log(f"🔍 [ERROR-DETECTED] System dialog visible for {sr_id}")
                return True
        
        # Method 3: Check for other common error indicators
        error_selectors = [
            "[class*='error']",
            "[class*='invalid']", 
            "[class*='alert']",
            ".error-message",
            ".validation-error"
        ]
        
        for selector in error_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if await element.is_visible():
                        log(f"🔍 [ERROR-DETECTED] Error element found: {selector} for {sr_id}")
                        return True
            except:
                continue  # Skip if selector fails
        
        # No errors detected
        return False
        
    except Exception as e:
        log(f"⚠️ [ERROR-CHECK] Exception during error detection for {sr_id}: {str(e)}")
        # If we can't check for errors, assume no error to continue
        return False


async def _handle_both_guesses_failed(sr_id: str, zzlocation: str) -> None:
    """Handle the case where both location guesses failed."""
    try:
        async with location_lock:
            if sr_id not in location_issue_tickets:
                ticket_data = {
                    "ticketid": sr_id, 
                    "zzlocation": zzlocation, 
                    "location": "", 
                    "description": "Both primary and zone C location guesses failed"
                }
                pending_location_requests[sr_id] = ticket_data
                awaiting_location_response[sr_id] = True
                location_issue_tickets.add(sr_id)
                
                # Send WhatsApp assistance request
                try:
                    send_location_request_whatsapp(ticket_data)
                    log(f"📞 [ASSIST] Location assistance requested for {sr_id}")
                except Exception as whatsapp_error:
                    log(f"⚠️ [WHATSAPP] Failed to send assistance request: {str(whatsapp_error)}")
                
    except Exception as e:
        log(f"⚠️ [ASSIST-ERROR] Failed to handle location failure for {sr_id}: {str(e)}")
        # Don't re-raise here - we still want to raise the main exception

def write_wo_output(sr: str, wo: str):
    with threading.Lock():
        df = pd.DataFrame([[sr,wo,datetime.now()]], columns=["SR","WO","Timestamp"])
        if os.path.exists(WO_OUTPUT_FILE):
            existing = pd.read_excel(WO_OUTPUT_FILE)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined.to_excel(WO_OUTPUT_FILE, index=False)
    log(f"💾 WO_Output.xlsx ← {sr},{wo}")

async def route_wo_and_extract(page, sr: str, wo: str) -> dict:
    # login
    await page.goto(f"{MAXIMO_BASE}/webclient/login/login.jsp")
    await page.fill("#username", USERNAME)
    await page.fill("#password", PASSWORD)
    await page.press("#password", "Enter")
    await page.wait_for_timeout(2000)

    # open WO tracking
    await page.goto(f"{MAXIMO_BASE}/ui/?event=loadapp&value=wotrack")
    await page.fill("#quicksearch", wo)
    await page.press("#quicksearch", "Enter")
    await wait_for_spinner(page)

    await page.wait_for_selector("#md3801d08-tb", timeout=10000)
    status= await page.input_value("#md3801d08-tb")
    if "WAPPR" in status:
        # insert your workorder field
        await page.fill("#mfdcef873-tb", "701740")
        await page.click("#ROUTEWF_MAIN-WO_-tbb_anchor")
        await wait_for_spinner(page)

        # confirm routing
        await page.wait_for_selector("#inputwf-dialog_inner", timeout=15000)
        await page.click("#m37917b04-pb")
        await wait_for_spinner(page)

    details = ""
    # try to extract details
    try:
        iframe_element = await page.query_selector("iframe[id$='-rte_iframe']")
        frame = await iframe_element.content_frame()
        await frame.wait_for_selector("#dijitEditorBody")
        details = await frame.inner_text("#dijitEditorBody")
    except Exception:
        details = ""  # fallback if iframe is missing or empty

    # build fields dictionary
    fields = {
        "SR":           sr,
        "WO":           wo,
        "location":     await page.input_value("#m7b0033b9-tb"),
        "type":         await page.input_value("#m41671ba1-tb"),
        "start_date":   await page.input_value("#m651c06b0-tb"),
        "finish_date":  await page.input_value("#mfc15570a-tb"),
        "report_date":  await page.input_value("#m972c8009-tb"),
        "description":  await page.input_value("#mad3161b5-tb2"),
        "priority":   await page.input_value("#m950e5295-tb"),
        "name": await page.input_value("#mdbe28d05-tb"),
        "timestamp_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # add details only if non-empty and not just whitespace
    if details.strip():
        fields["details"] = details.strip()

    return fields,details

def write_details(rec: dict):
    with threading.Lock():
        df = pd.DataFrame([rec])
        if os.path.exists(DETAILS_OUTPUT_FILE):
            existing = pd.read_excel(DETAILS_OUTPUT_FILE)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df

        combined.to_excel(DETAILS_OUTPUT_FILE, index=False)
    log(f"💾 SR_WO_Details.xlsx ← {rec}")

def send_whatsapp_message(group_id, message):
    url = "http://localhost:3000/send-message"
    data = {
        "groupId": group_id,
        "message": message
    }
    try:
        res = requests.post(url, json=data)
        log(res.text)
    except Exception as e:
        log(f"Failed to send WhatsApp message: {e}")

def fetch_recent_queued_tickets_with_retry(session, max_retries=3) -> List[Dict]:
    """Fetch tickets with retry logic"""
    for attempt in range(max_retries):
        try:
            return fetch_recent_queued_tickets(session)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            log(f"⚠️ Network error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                log("❌ All retry attempts failed")
                return []
        except Exception as e:
            log(f"❌ Unexpected error fetching tickets: {e}")
            return []
    return []

def fetch_recent_queued_tickets(session) -> List[Dict]:
    ts = (datetime.now(timezone.utc) - timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "oslc.where": f'status="QUEUED" and reportdate>="{ts}" and siteid="OJ"',
        "oslc.select": "zzlocation,ticketid,reportedby,description,status,siteid,reportdate,location",
        "lean": 1
    }

    try:
        response = session.get(TICKET_FETCH_URL, params=params, timeout=30)
        response.raise_for_status()  # ADD THIS LINE
        if response.status_code == 200:
            data = response.json()
            return data.get("member", [])
        else:
            log(f"❌ [ERROR] Failed to fetch tickets: {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        log("❌ [ERROR] Request timeout while fetching tickets")
        return []
    except requests.exceptions.ConnectionError:
        log("❌ [ERROR] Connection error while fetching tickets")
        return []
    except Exception as e:
        log(f"❌ [ERROR] Failed to fetch tickets: {e}")
        return []

async def producer():
    global step1_workers_started, step2_workers_started, location_issue_tickets, step1_guess_started
    fmt = re.compile(r"^\s*\d+\s*[-_]\s*\d+\s*$")
    
    consecutive_failures = 0
    max_failures = 3

    while is_automation_running() and not await check_shutdown():
        session = None
        try:
            # Create fresh session for each iteration
            session = create_session()
            
            if ticket_queue.qsize() == 0:
                log("🔄 Checking for new tickets...")
                
                # Add timeout wrapper
                try:
                    incoming_tickets = await asyncio.wait_for(
                        asyncio.to_thread(fetch_recent_queued_tickets_with_retry, session), 
                        timeout=45  # 45 second timeout
                    )
                    consecutive_failures = 0  # Reset on success
                except asyncio.TimeoutError:
                    log("⚠️ Ticket fetch timeout, retrying...")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        log("❌ Too many consecutive failures, waiting longer...")
                        await asyncio.sleep(300)  # Wait 5 minutes
                        consecutive_failures = 0
                    continue
                except Exception as e:
                    log(f"❌ Error fetching tickets: {e}")
                    consecutive_failures += 1
                    continue

                # Process tickets logic (same as before)
                valid_tickets = []
                for ticket in incoming_tickets:
                    if await check_shutdown():
                        break
                        
                    ticketid = ticket["ticketid"]
                    location = ticket.get("location", "").strip()
                    description = ticket.get("description", "").lower()
                    zz  = ticket.get("zzlocation","")
                    
                    log(f"ticketid : {ticketid} , location : {location} , zzlocation :{zz} ,  description : {description}")
                    if "702579" not in ticket.get("reportedby") and "701740" not in ticket.get("reportedby"):
                        if "elevator" in description:
                            continue
                        
                        async with location_lock:
                            if ticketid in location_issue_tickets:
                                continue

                        if is_valid_loc(location):
                            valid_tickets.append(ticket)
                        elif fmt.match(zz):
                            if ticketid not in processing_tickets:
                                await guess_queue.put(ticket)
                        else:
                            location_issue_tickets.add(ticketid)
                            send_location_request_whatsapp(ticket)
                            awaiting_location_response[ticketid] = True
                            pending_location_requests[ticketid] = ticket
                            log(f"📍 Location assistance requested for {ticketid} (invalid guess)")
                            body = "🚫 The following SR has invalid location:\n\n"
                            body += f"- SR: {ticketid} | Location: {location} | Description: {description}\n"
                            send_email("Invalid SR Location Detected", body)
                            log(f"📧 Reported {ticketid} invalid SR location")
                    else:
                        valid_tickets.append(ticket)

                # In producer(), replace the current duplicate check:
                for ticket in valid_tickets:
                    if await check_shutdown():
                        break
                    ticketid = ticket["ticketid"]
                    
                    # Check both queue and currently processing tickets
                    with processing_lock:
                        already_queued = any(ticketid == existing["ticketid"] for existing in ticket_queue._queue)
                        currently_processing = ticketid in processing_tickets
                    
                    if not already_queued and not currently_processing:
                        await ticket_queue.put(ticket)
                        log(f"➕ Ticket queued: {ticketid}")
            # Start workers if needed
            if not ticket_queue.empty() and not step1_workers_started and is_automation_running():
                step1_workers_started = True
                log(f"🔧 Starting Step 1 workers (queue size: {ticket_queue.qsize()})")
                for i in range(NUM_WORKERS):
                    asyncio.create_task(step1_worker(i))

            if not guess_queue.empty() and not step1_guess_started and is_automation_running():
                step1_guess_started = True
                log(f"🔧 Starting Step 1 Guess workers (queue size: {guess_queue.qsize()})")
                for i in range(NUM_WORKERS):
                    asyncio.create_task(step1_guess_worker(i))

            # Shorter wait time with frequent shutdown checks
            for _ in range(60):  # 30 seconds instead of 120
                if await check_shutdown():
                    break
                await asyncio.sleep(1)
                
        except Exception as e:
            log(f"❌ Producer error: {e}")
            consecutive_failures += 1
            if await check_shutdown():
                break
            await asyncio.sleep(10)  # Wait before retrying
        finally:
            # Close session properly
            if session:
                try:
                    session.close()
                except:
                    pass

    log("🛑 Producer stopped")

async def step1_worker(wid: int):
    global active_step1_workers, step1_workers_started
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        active_step1_workers += 1
        try:
            while is_automation_running() and not await check_shutdown():
                try:
                    sr_rec = await asyncio.wait_for(ticket_queue.get(), timeout=5.0)
                    sr = sr_rec["ticketid"]
                    expected_location = sr_rec.get("location", "").strip()  # NEW: Get expected location
                    
                    # Mark as being processed ONCE, outside the retry loop
                    with processing_lock:
                        processing_tickets.add(sr)
                        
                except asyncio.TimeoutError:
                    if ticket_queue.empty():
                        break
                    continue
                    
                for attempt in range(1, RETRY_LIMIT+1):
                    if await check_shutdown():
                        break
                        
                    context = await browser.new_context()
                    page = await context.new_page()
                    try:
                        log(f"[S1][W{wid}] SR→WO {sr} try {attempt}")
                        # MODIFIED: Pass expected location to the function
                        wo = await route_sr_and_get_wo(page, sr, expected_location if expected_location else None)
                        log(f"[S1][W{wid}] ✅ {sr}→{wo}")
                        write_wo_output(sr, wo)
                        await wo_queue.put((sr, wo))
                        
                        # INCREMENT THE COUNTER HERE!
                        count = increment_processed_tickets()
                        log(f"📈 Processed tickets count: {count}")
                        
                        break
                    except Exception as e:
                        log(f"[S1][W{wid}] ⚠️ {sr} attempt {attempt}: {e}")
                        if attempt == RETRY_LIMIT:
                            send_email(f"SR routing failed {sr}", str(e))
                    finally:
                        await page.close()
                        await context.close()
                        await asyncio.sleep(1)
                
                # Remove from processing ONCE, after all retry attempts
                with processing_lock:
                    processing_tickets.discard(sr)
                ticket_queue.task_done()
        finally:
            active_step1_workers -= 1
            if active_step1_workers == 0:
                step1_workers_started = False
                log(f"🔄 All Step1 workers stopped, flag reset")
            await browser.close()
            log(f"🛑 Step1 Worker {wid} stopped")
# Fix 2: Update step2_worker function  
async def step2_worker(wid: int):
    global active_step2_workers, step2_workers_started
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        active_step2_workers += 1
        try:
            # Keep running while automation is active AND queue has items
            while is_automation_running() and not await check_shutdown():
                try:
                    # Use timeout to prevent hanging
                    sr, wo = await asyncio.wait_for(wo_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Check if there are more items or if we should continue
                    if wo_queue.empty():
                        break
                    continue
                    
                for attempt in range(1, RETRY_LIMIT+1):
                    if await check_shutdown():
                        break
                        
                    context = await browser.new_context()
                    page = await context.new_page()
                    try:
                        log(f"[S2][W{wid}] WO extract {wo} try {attempt}")
                        rec,details = await route_wo_and_extract(page, sr, wo)
                        log(f"[S2][W{wid}] ✅ extracted {wo}")
                        write_details(rec)
                        try:
                            add_row_to_google_sheet_with_formatting(rec,use_formatting=True)
                        except Exception as e:
                            log(f"⚠️ Google Sheets update failed for {sr}: {e}")
                        msg = f"""\n🛠 SR {sr} → WO {wo} @ {rec['timestamp_date']}\n\n🔹 Location: {rec['location']}\n🔹 Type: {rec['type']}\n🗓 Start: {rec['start_date']} | Finish: {rec['finish_date']}\n🔹 Reported: {rec['report_date']}\n🔹 Description: {rec['description']}\n🔹 priority: {rec['priority']}\n"""
                        if details and details.strip():
                            msg += f"📝 Details:\n{details.strip()}\n"
                        name=rec['name']
                        if name and name.strip():
                            msg+=f"\n\n!------{rec['name']}------!"
                        send_whatsapp(msg)
                        break
                    except Exception as e:
                        log(f"[S2][W{wid}] ⚠️ {wo} attempt {attempt}: {e}")
                        if attempt == RETRY_LIMIT:
                            send_email(f"WO extract failed {wo}", str(e))
                    finally:
                        await page.close()
                        await context.close()
                        await asyncio.sleep(1)
                wo_queue.task_done()
        finally:
            active_step2_workers -= 1
            # Reset flag only when ALL workers are done
            if active_step2_workers == 0:
                step2_workers_started = False
                log(f"🔄 All Step2 workers stopped, flag reset")
            await browser.close()
            log(f"🛑 Step2 Worker {wid} stopped")

async def step1_guess_worker(wid):
    global active1g, step1_guess_started
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        active1g += 1
        try:
            while is_automation_running() and not await check_shutdown():
                try:
                    t = await asyncio.wait_for(guess_queue.get(), timeout=5.0)
                    ticketid = t["ticketid"]
                    
                    # Mark as being processed ONCE, outside the retry loop
                    with processing_lock:
                        processing_tickets.add(ticketid)
                        
                except asyncio.TimeoutError:
                    if guess_queue.empty():
                        break
                    continue
                    
                for attempt in range(1, RETRY_LIMIT+1):
                    if await check_shutdown():
                        break
                        
                    context = await browser.new_context()
                    page = await context.new_page()
                    try:
                        log(f"[S1-G][W{wid}] SR→WO+GUESS {t['ticketid']} try {attempt}")
                        wo = await route_sr_and_get_wo_with_guess(page, t["ticketid"], t["zzlocation"])
                        write_wo_output(t["ticketid"], wo)
                        await wo_queue.put((t["ticketid"], wo))
                        
                        # INCREMENT THE COUNTER HERE TOO!
                        count = increment_processed_tickets()
                        log(f"📈 Processed tickets count: {count}")
                        
                        break
                    except Exception as e:
                        log(f"[S1-G][W{wid}] ⚠️ {e}")
                        if attempt == RETRY_LIMIT:
                            send_email("SR routing+guess failed", str(e))
                    finally:
                        await page.close()
                        await context.close()
                        await asyncio.sleep(1)
                
                # Remove from processing ONCE, after all retry attempts
                with processing_lock:
                    processing_tickets.discard(ticketid)
                guess_queue.task_done()
        finally:
            active1g -= 1
            if active1g == 0:
                step1_guess_started = False
                log(f"🔄 All Step1 Guess workers stopped, flag reset")
            await browser.close()
            log(f"🛑 Step1 Guess Worker {wid} stopped")

# Fix 4: Update trigger_workers function
async def trigger_workers():
    global step2_workers_started

    # More robust check for starting Step 2 workers
    if not wo_queue.empty() and not step2_workers_started and is_automation_running():
        step2_workers_started = True
        log(f"🔧 Starting Step 2 workers (queue size: {wo_queue.qsize()})")
        for i in range(NUM_WORKERS):
            asyncio.create_task(step2_worker(i))
        log("🔧 Step 2 workers started")

async def monitor_loop():
    while is_automation_running() and not await check_shutdown():
        try:
            await trigger_workers()
            
            # Wait with interruption checking
            for _ in range(15):  # 15 seconds
                if await check_shutdown():
                    break
                await asyncio.sleep(1)
        except Exception as e:
            log(f"❌ Monitor loop error: {e}")
            if await check_shutdown():
                break
            await asyncio.sleep(5)
    
    log("🛑 Monitor loop stopped")

# ───── Main Function with Proper Initialization and Cleanup ─────
async def main(external_stop_event=None):
    """Main automation pipeline function"""
    global stop_event
    stop_event = external_stop_event  # Store the external stop event
    
    # Reset state
    global step1_workers_started, step2_workers_started, step1_guess_started
    global active_step1_workers, active_step2_workers, active1g
    
    set_automation_running(True)
    shutdown_event.clear()
    
    step1_workers_started = False
    step2_workers_started = False  
    step1_guess_started = False
    active_step1_workers = 0
    active_step2_workers = 0
    active1g = 0
    
    log("▶ Pipeline started")
    
    try:
        # Run the main automation components
        await asyncio.gather(
            producer(),
            monitor_loop(),
            return_exceptions=True
        )
    except Exception as e:
        log(f"❌ Main pipeline error: {e}")
    finally:
        # Cleanup
        set_automation_running(False)
        log("🛑 Pipeline stopped")

# ───── Control Functions for External Management ─────
def start_automation():
    """Start the automation pipeline"""
    if is_automation_running():
        return False, "Automation is already running"
    
    log("🚀 Starting automation pipeline...")
    return True, "Automation started"

def stop_automation():
    """Stop the automation pipeline"""
    if not is_automation_running():
        return False, "Automation is not running"
    
    log("🛑 Stopping automation pipeline...")
    signal_shutdown()
    return True, "Automation stop signal sent"

def get_automation_status():
    """Get current automation status"""
    return {
        "running": is_automation_running(),
        "workers": {
            "step1_active": active_step1_workers,
            "step2_active": active_step2_workers,
            "guess_active": active1g
        },
        "queues": {
            "tickets": ticket_queue.qsize() if ticket_queue else 0,
            "work_orders": wo_queue.qsize() if wo_queue else 0,
            "guess": guess_queue.qsize() if guess_queue else 0
        }
    }
# Add at the end of the file, before if __name__ == "__main__":
def handle_whatsapp_location_response(sr_id: str, response: str, location: str = None):
    """API endpoint to handle WhatsApp location responses"""
    try:
        handle_location_assistance_response(sr_id, response, location)
        return True, f"Response processed for SR {sr_id}"
    except Exception as e:
        log(f"❌ Error processing location response: {e}")
        return False, str(e)

if __name__== "__main__":
    asyncio.run(main())
