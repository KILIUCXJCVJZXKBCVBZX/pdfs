class DataAugmenter:
    """Augment training data to address sparsity issues"""
    
    def __init__(self):
        self.synonyms = {
            'broken': ['damaged', 'faulty', 'defective', 'malfunctioning'],
            'repair': ['fix', 'restore', 'mend', 'replace'],
            'not working': ['not functioning', 'inoperative', 'out of order', 'failed'],
            'leakage': ['leak', 'seepage', 'dripping', 'water escape'],
            'connection': ['joint', 'fitting', 'coupling', 'link'],
            'system': ['equipment', 'unit', 'device', 'apparatus'],
            'door': ['entrance', 'doorway', 'portal', 'access'],
            'lock': ['latch', 'fastener', 'security', 'mechanism'],
            'handle': ['grip', 'knob', 'lever', 'control']
        }
    
    def paraphrase_text(self, text: str, num_variations: int = 2) -> List[str]:
        """Generate paraphrased versions of text"""
        variations = [text]  # Original text
        
        for _ in range(num_variations):
            new_text = text.lower()
            
            # Replace with synonyms
            for word, syns in self.synonyms.items():
                if word in new_text:
                    new_text = new_text.replace(word, np.random.choice(syns))
            
            # Minor variations
            new_text = re.sub(r'\b(the|a|an)\b', '', new_text)  # Remove articles
            new_text = re.sub(r'\s+', ' ', new_text).strip()  # Clean spaces
            
            if new_text != text.lower() and new_text not in variations:
                variations.append(new_text)
        
        return variations
    
    def augment_dataset(self, df: pd.DataFrame, min_samples_per_path: int = 50) -> pd.DataFrame:
        """Augment dataset for underrepresented paths"""
        # Ensure combined_text column exists
        if 'combined_text' not in df.columns:
            raise ValueError("DataFrame must have 'combined_text' column. Call prepare_combined_text() first.")
        
        # Count samples per complete path
        path_counts = df.groupby(['failure_class', 'problem', 'cause', 'remedy']).size()
        
        augmented_rows = []
        
        for path, count in path_counts.items():
            if count < min_samples_per_path:
                # Get original rows for this path
                mask = ((df['failure_class'] == path[0]) & 
                       (df['problem'] == path[1]) & 
                       (df['cause'] == path[2]) & 
                       (df['remedy'] == path[3]))
                
                original_rows = df[mask]
                
                # Generate additional samples
                needed = min_samples_per_path - count
                
                for _, row in original_rows.iterrows():
                    # Generate variations of combined text (description + remark)
                    combined_text = row['combined_text']
                    text_variations = self.paraphrase_text(combined_text, 
                                                         needed // len(original_rows) + 1)
                    
                    for i, new_text in enumerate(text_variations[1:]):  # Skip original
                        if len(augmented_rows) >= needed:
                            break
                        
                        new_row = row.copy()
                        new_row['combined_text'] = new_text
                        new_row['work_order_id'] = f"{row['work_order_id']}_aug_{i}"
                        augmented_rows.append(new_row)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

class FeatureEngineer:
    """Enhanced feature engineering for Maximo data with combined text"""
    
    def __init__(self):
        self.vectorizers = {}
        self.domain_keywords = [
            'electrical', 'plumbing', 'hvac', 'lighting', 'mechanical',
            'leak', 'broken', 'repair', 'replace', 'fix', 'damage',
            'connection', 'system', 'equipment', 'maintenance',
            'door', 'lock', 'handle', 'key', 'closure', 'sliding',
            'carpentry', 'civil services', 'issue', 'found', 'reported',
            'building', 'apartment', 'unit', 'floor', 'main'
        ]
    
    def extract_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """Extract comprehensive features from combined text"""
        
        # TF-IDF with domain-specific terms - enhanced for maintenance context
        if fit or 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1500,  # Increased for richer text
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.8,
                lowercase=True,
                strip_accents='ascii'
            )
            tfidf_features = self.vectorizers['tfidf'].fit_transform(texts)
        else:
            tfidf_features = self.vectorizers['tfidf'].transform(texts)
        
        # Domain keyword features
        keyword_features = []
        for text in texts:
            text_lower = text.lower()
            features = [1 if keyword in text_lower else 0 for keyword in self.domain_keywords]
            keyword_features.append(features)
        
        keyword_features = np.array(keyword_features)
        
        # Text length and pattern features
        text_features = []
        for text in texts:
            features = [
                len(text),  # Text length
                len(text.split()),  # Word count
                text.count('|'),  # Separator count (if combining desc + remark)
                1 if any(char.isdigit() for char in text) else 0,  # Contains numbers
                1 if re.search(r'\b(urgent|emergency|asap)\b', text.lower()) else 0,  # Urgency
            ]
            text_features.append(features)
        
        text_features = np.array(text_features)
        
        # FIXED: Ensure all arrays have the same number of rows (samples)
        n_samples = len(texts)
        
        # Convert TF-IDF to dense array
        tfidf_dense = tfidf_features.toarray()
        
        # Ensure keyword_features and text_features have correct shape
        if keyword_features.shape[0] != n_samples:
            print(f"Warning: keyword_features shape mismatch. Expected {n_samples}, got {keyword_features.shape[0]}")
            keyword_features = keyword_features[:n_samples] if keyword_features.shape[0] > n_samples else keyword_features
        
        if text_features.shape[0] != n_samples:
            print(f"Warning: text_features shape mismatch. Expected {n_samples}, got {text_features.shape[0]}")
            text_features = text_features[:n_samples] if text_features.shape[0] > n_samples else text_features
        
        # Debug: Print shapes before concatenation
        print(f"Feature shapes - TF-IDF: {tfidf_dense.shape}, Keywords: {keyword_features.shape}, Text: {text_features.shape}")
        
        # Combine all features - FIXED: Only include the defined features
        combined_features = np.hstack([
            tfidf_dense, 
            keyword_features,
            text_features
        ])
        
        print(f"Combined features shape: {combined_features.shape}")
        
        return combined_features
def prepare_combined_text(df: pd.DataFrame, desc_col: str = 'description', remark_col: str = 'remark') -> pd.DataFrame:
    """Prepare combined text from description and remark columns"""
    df = df.copy()
    
    # Handle missing values
    df[desc_col] = df[desc_col].fillna('').astype(str)
    df[remark_col] = df[remark_col].fillna('').astype(str)
    
    # Combine description and remark with separator
    # Use ' | ' as separator to help model distinguish between the two parts
    df['combined_text'] = df[desc_col] + ' | ' + df[remark_col]
    
    # Clean up the combined text
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces
    df['combined_text'] = df['combined_text'].str.replace(' | ', ' | ', regex=False)  # Clean separator
    df['combined_text'] = df['combined_text'].str.strip()  # Trim whitespace
    
    # Remove cases where combined text is just the separator
    df['combined_text'] = df['combined_text'].replace('|', '').str.strip()
    df['combined_text'] = df['combined_text'].replace('', 'No description available')
    
    print(f"✅ Combined text prepared. Average length: {df['combined_text'].str.len().mean():.1f} characters")
    print(f"Sample combined text: '{df['combined_text'].iloc[0][:100]}...'")
    
    return df

class EnhancedHierarchicalClassifier:
    """Enhanced hierarchical classifier with combined text support"""
    
    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble
        self.models = {}
        self.label_encoders = {}
        self.feature_engineer = FeatureEngineer()
        self.feedback_data = []  # Store user feedback
        self.pattern_weights = {}  # Weight patterns based on feedback
        
        # Dual hierarchy approach
        self.training_hierarchy = {}  # Built from training data
        self.full_hierarchy = {}     # Complete Maximo hierarchy (fallback)
        self.path_frequencies = {}   # Track frequency of paths in training

        self.concept_mappings = {
            'plumbing_concepts': {
                'keywords': ['water', 'heater', 'pipe', 'leak', 'plumbing', 'supply', 'drain', 
                           'faucet', 'toilet', 'shower', 'hot water', 'cold water', 'pressure'],
                'failure_classes': ['PLUMBING', 'Pipe', 'Water Supply'],
                'problems': ['Water heater not working', 'Leakage', 'Water Leak', 'No Hot Water'],
                'causes': ['Water heater defected', 'Leakage In System', 'Pipe Damage'],
                'remedies': ['Replace', 'Repair/Replace', 'Fix Leak']
            },
            'electrical_concepts': {
                'keywords': ['power', 'electrical', 'circuit', 'breaker', 'wiring', 'voltage',
                           'switch', 'outlet', 'lighting', 'heater'],
                'failure_classes': ['CIRCUIT BREAKER', 'Heater', 'Lighting Systems'],
                'problems': ['Not Working', 'Unable To Reset', 'Deformation'],
                'causes': ['No Power', 'Overloaded', 'Loose Connections'],
                'remedies': ['Check Incoming Feeder', 'Check Load Side', 'Replace & tighten']
            },
            'civil_concepts': {
                'keywords': ['door', 'lock', 'handle', 'key', 'entrance', 'carpentry', 'civil'],
                'failure_classes': ['Civil Services'],
                'problems': ['Doors', 'NOT WORKING'],
                'causes': ['Lock Damage', 'Handle Damage'],
                'remedies': ['Replace', 'REPAIR/FIX']
            }
        }
        
        # ADD: Concept similarity weights
        self.concept_similarity_boost = 0.3  # Boost for matching concepts
        
        self.confidence_threshold = 0.7
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        

    def add_feedback(self, description: str, remark: str, correct_path: List[str], 
                    confidence_boost: float = 0.2):
        """Enhanced feedback addition with separate description and remark storage"""
        # Clean inputs
        desc_clean = str(description).strip() if description else ""
        remark_clean = str(remark).strip() if remark else ""
        
        # Create combined text for compatibility
        if remark_clean:
            combined_text = f"{desc_clean} | {remark_clean}"
        else:
            combined_text = desc_clean
        
        # Clean up combined text
        combined_text = combined_text.replace('  ', ' ').strip()
        if combined_text.endswith('|'):
            combined_text = combined_text[:-1].strip()
        
        feedback_entry = {
            'text': combined_text,
            'description': desc_clean,  # ADD: Store separately
            'remark': remark_clean,     # ADD: Store separately
            'correct_path': correct_path,
            'timestamp': datetime.now().isoformat(),
            'confidence_boost': confidence_boost,
            'original_description': description,
            'original_remark': remark
        }
        
        self.feedback_data.append(feedback_entry)
        
        # MODIFY: Update pattern weights with new logic
        self._update_pattern_weights_restricted(desc_clean, remark_clean, correct_path, confidence_boost)
        
        self.logger.info(f"Added feedback for: '{combined_text[:50]}...'")
        self.logger.info(f"Correct path: {correct_path}")
        self.logger.info(f"Total feedback entries: {len(self.feedback_data)}")

    def _get_intelligent_fallback(self, level: str, parent_path: List[str], text: str) -> str:
        """Enhanced fallback using semantic concept analysis"""
        
        # Get basic valid options
        valid_options = self._get_valid_options(level, parent_path)
        
        if not valid_options:
            return self._get_basic_fallback(level)
        
        # Analyze text for concept matches
        text_lower = text.lower()
        option_scores = {}
        
        for option in valid_options:
            score = 0.0
            
            # Score based on concept matching
            for concept_name, concept_data in self.concept_mappings.items():
                # Check if text matches this concept domain
                text_concept_score = sum(1 for keyword in concept_data['keywords'] 
                                    if keyword in text_lower)
                
                if text_concept_score > 0:  # Text relates to this concept
                    # Check if option is in this concept's vocabulary
                    level_options = concept_data.get(f'{level}s', [])  # problems -> problems
                    if level == 'failure_class':
                        level_options = concept_data.get('failure_classes', [])
                    elif level == 'cause':
                        level_options = concept_data.get('causes', [])
                    elif level == 'remedy':
                        level_options = concept_data.get('remedies', [])
                    
                    if option in level_options:
                        score += text_concept_score * self.concept_similarity_boost
            
            # Add frequency-based score (existing logic)
            if self.path_frequencies:
                for path, freq in self.path_frequencies.items():
                    if self._path_matches(path, parent_path + [option], level):
                        score += freq * 0.01  # Small frequency boost
            
            option_scores[option] = score
        
        # Return option with highest score, or first if all zeros
        if max(option_scores.values()) > 0:
            return max(option_scores.items(), key=lambda x: x[1])[0]
        else:
            return valid_options[0]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity percentage between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity calculation
        matcher = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip())
        return matcher.ratio()   
    def _get_default_remedies(self, failure_class: str, problem: str, cause: str) -> List[str]:
        """Get default remedies based on failure class, problem, and cause"""
        default_remedies = ['REPAIR', 'REPLACE', 'MAINTAIN', 'INSPECT']
        
        # Add some basic logic based on keywords
        if 'electrical' in failure_class.lower():
            default_remedies.extend(['REWIRE', 'CHECK_CONNECTIONS'])
        elif 'plumbing' in failure_class.lower():
            default_remedies.extend(['SEAL_LEAK', 'REPLACE_PIPE'])
        elif 'mechanical' in failure_class.lower():
            default_remedies.extend(['LUBRICATE', 'ADJUST'])
        elif 'civil' in failure_class.lower():
            default_remedies.extend(['FIX', 'REPAIR_STRUCTURE'])
            
        return list(set(default_remedies))  # Remove duplicates
        
    def _update_pattern_weights_restricted(self, description: str, remark: str, correct_path: List[str], boost: float):
        # CHANGE: Store with correct composite key structure
        pattern_key = (description.lower().strip(), remark.lower().strip() if remark else "")
        path_key = tuple(correct_path)
        composite_key = (pattern_key, path_key)
        
        # CHANGE: Store weight directly (not nested dict)
        self.pattern_weights[composite_key] = self.pattern_weights.get(composite_key, 0.0) + boost

    def _apply_feedback_weights(self, text: str, predictions: Dict) -> Dict:
        """Apply feedback weights with restricted matching (exact description + 40% remark similarity)"""
        # Parse input text to extract description and remark
        input_desc, input_remark = self._parse_combined_text(text)
        
        self.logger.info(f"Applying restricted feedback matching")
        self.logger.info(f"Input description: '{input_desc}'")
        self.logger.info(f"Input remark: '{input_remark}'")
        
        # Apply feedback to each path
        if 'top_paths' in predictions:
            for path_result in predictions['top_paths']:
                path_tuple = tuple(path_result['path'])
                
                # Calculate boost for this path using restricted matching
                total_boost = 0.0
                matched_patterns = []
                
                for composite_key, weight in self.pattern_weights.items():
                    (pattern_key, stored_path) = composite_key
                    stored_desc, stored_remark = pattern_key
    
    
                    if stored_path == path_tuple:
                        # Check exact description match
                        if stored_desc == input_desc.lower().strip():
                            # Check remark similarity if both exist
                            remark_match = True
                            if stored_remark and input_remark:
                                similarity = self._calculate_similarity(stored_remark, input_remark)
                                remark_match = similarity >= 0.4  # 40% threshold
                                self.logger.info(f"Remark similarity: {similarity:.3f} ({'✓' if remark_match else '✗'})")
                            elif not stored_remark and not input_remark:
                                remark_match = True  # Both empty
                            else:
                                remark_match = False  # One empty, other not
                            
                            if remark_match:
                                weight = weight [path_tuple]
                                total_boost += weight
                                matched_patterns.append((f"desc_exact:{stored_desc[:20]}...", weight))
                                self.logger.info(f"✓ Exact match found - applying boost: {weight}")
                
                # Apply boost
                if total_boost > 0:
                    old_confidence = path_result['overall_confidence']
                    path_result['overall_confidence'] = min(1.0, old_confidence + total_boost * 1.5)
                    path_result['feedback_boost'] = total_boost * 1.5
                    path_result['matched_patterns'] = matched_patterns
                    
                    self.logger.info(f"Applied restricted feedback to path {path_tuple}: {old_confidence:.3f} -> {path_result['overall_confidence']:.3f}")
                else:
                    path_result['feedback_boost'] = 0.0
                    path_result['matched_patterns'] = []
            
            # Re-sort by confidence
            predictions['top_paths'].sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        return predictions
    
    def _parse_combined_text(self, combined_text: str) -> Tuple[str, str]:
        """Parse combined text to extract description and remark"""
        if '|' in combined_text:
            parts = combined_text.split('|', 1)
            description = parts[0].strip()
            remark = parts[1].strip() if len(parts) > 1 else ""
        else:
            description = combined_text.strip()
            remark = ""
        
        return description, remark

    def _enhance_feature_extraction_with_patterns(self):
        """Add plumbing-specific patterns to feature engineer"""
        plumbing_patterns = [
            'water supply', 'water leak', 'water heater', 'plumbing system',
            'pipe', 'faucet', 'drain', 'toilet', 'shower', 'bath',
            'hot water', 'cold water', 'pressure', 'flow'
        ]
        
        electrical_patterns = [
            'heater', 'power', 'electrical', 'wiring', 'outlet', 
            'switch', 'breaker', 'voltage', 'current'
        ]
        
        # Add to feature engineer
        self.feature_engineer.domain_keywords.extend(plumbing_patterns)
        self.feature_engineer.domain_keywords.extend(electrical_patterns)
        
        # Create pattern mappings
        self.pattern_to_failure_class = {
            'plumbing': plumbing_patterns,
            'electrical': electrical_patterns
        }


    
    def load_full_hierarchy(self, hierarchy_data: pd.DataFrame):
        """Load complete Maximo hierarchy from DataFrame"""
        self.logger.info("Loading full Maximo hierarchy...")
        
        # Print available columns for debugging
        self.logger.info(f"Available columns in hierarchy file: {list(hierarchy_data.columns)}")
        
        hierarchy = {}
        
        # Check what columns are actually available
        available_cols = hierarchy_data.columns.tolist()
        
        # Method 1: Direct hierarchy structure
        hierarchy_cols = ['failure_class', 'problem', 'cause', 'remedy']
        
        if all(col in available_cols for col in hierarchy_cols):
            self.logger.info("Using direct hierarchy columns approach")
            
            for _, row in hierarchy_data.iterrows():
                fc = str(row['failure_class']).strip()
                prob = str(row['problem']).strip()
                cause = str(row['cause']).strip()
                remedy = str(row['remedy']).strip()
                
                # Skip empty rows
                if not fc or fc.lower() in ['nan', 'none', '']:
                    continue
                    
                if fc not in hierarchy:
                    hierarchy[fc] = {}
                if prob not in hierarchy[fc]:
                    hierarchy[fc][prob] = {}
                if cause not in hierarchy[fc][prob]:
                    hierarchy[fc][prob][cause] = set()
                
                hierarchy[fc][prob][cause].add(remedy)
        
        # Method 2: Build from available structure
        elif 'failure_class' in available_cols and 'problem' in available_cols:
            self.logger.info("Using available structure approach")
            
            # Get unique combinations to build hierarchy
            required_cols = [col for col in hierarchy_cols if col in available_cols]
            unique_combinations = hierarchy_data[required_cols].drop_duplicates()
            
            for _, row in unique_combinations.iterrows():
                fc = str(row['failure_class']).strip()
                prob = str(row.get('problem', 'GENERAL PROBLEM')).strip()
                cause = str(row.get('cause', 'UNKNOWN')).strip()
                
                # Skip empty rows
                if not fc or fc.lower() in ['nan', 'none', '']:
                    continue
                    
                if fc not in hierarchy:
                    hierarchy[fc] = {}
                if prob not in hierarchy[fc]:
                    hierarchy[fc][prob] = {}
                if cause not in hierarchy[fc][prob]:
                    hierarchy[fc][prob][cause] = set()
                
                # For remedies, use default or from data
                if 'remedy' in row:
                    remedy = str(row['remedy']).strip()
                    if remedy and remedy.lower() not in ['nan', 'none', '']:
                        hierarchy[fc][prob][cause].add(remedy)
                
                # Add default remedies
                default_remedies = self._get_default_remedies(fc, prob, cause)
                for remedy in default_remedies:
                    hierarchy[fc][prob][cause].add(remedy)
        
        # Convert sets to lists for pickle compatibility
        for fc in hierarchy:
            for prob in hierarchy[fc]:
                for cause in hierarchy[fc][prob]:
                    hierarchy[fc][prob][cause] = list(hierarchy[fc][prob][cause])
        
        self.full_hierarchy = hierarchy
        self.logger.info(f"Loaded full hierarchy with {len(self.full_hierarchy)} failure classes")
    
    def _create_model(self, model_type: str = 'ensemble'):
        """Create model based on type"""
        if model_type == 'ensemble' and self.use_ensemble:
            return VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('svm', SVC(probability=True, random_state=42))
            ], voting='soft')
        elif model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LogisticRegression(random_state=42, max_iter=1000)
    
    def _build_training_hierarchy(self, df: pd.DataFrame):
        """Build hierarchy from training data with frequency tracking"""
        hierarchy = {}
        frequencies = {}
        
        for _, row in df.iterrows():
            fc, prob, cause, remedy = row['failure_class'], row['problem'], row['cause'], row['remedy']
            path = (fc, prob, cause, remedy)
            
            # Build hierarchy structure
            if fc not in hierarchy:
                hierarchy[fc] = {}
            if prob not in hierarchy[fc]:
                hierarchy[fc][prob] = {}
            if cause not in hierarchy[fc][prob]:
                hierarchy[fc][prob][cause] = set()
            
            hierarchy[fc][prob][cause].add(remedy)
            
            # Track frequencies
            frequencies[path] = frequencies.get(path, 0) + 1
        
        # Convert sets to lists
        for fc in hierarchy:
            for prob in hierarchy[fc]:
                for cause in hierarchy[fc][prob]:
                    hierarchy[fc][prob][cause] = list(hierarchy[fc][prob][cause])
        
        self.training_hierarchy = hierarchy
        self.path_frequencies = frequencies
        self.logger.info(f"Built training hierarchy with {len(self.training_hierarchy)} failure classes")
    
    def _get_valid_options(self, level: str, parent_path: List[str]) -> List[str]:
        """Get valid options for a level given parent path - hybrid approach"""
        options = []
        
        # First try training hierarchy (preferred - based on actual data)
        training_options = self._get_options_from_hierarchy(self.training_hierarchy, level, parent_path)
        if training_options:
            options.extend(training_options)
        
        # Then supplement with full hierarchy if training data is insufficient
        if self.full_hierarchy and (not options or len(options) < 3):
            full_options = self._get_options_from_hierarchy(self.full_hierarchy, level, parent_path)
            if full_options:
                # Add options not already in training options
                for option in full_options:
                    if option not in options:
                        options.append(option)
        
        return list(set(options))  # Remove duplicates
    
    def _get_options_from_hierarchy(self, hierarchy: dict, level: str, parent_path: List[str]) -> List[str]:
        """Extract options from specific hierarchy"""
        if not hierarchy or not parent_path:
            return []
        
        try:
            if level == 'problem' and len(parent_path) >= 1:
                fc = parent_path[0]
                return list(hierarchy.get(fc, {}).keys())
            
            elif level == 'cause' and len(parent_path) >= 2:
                fc, prob = parent_path[0], parent_path[1]
                if fc in hierarchy and prob in hierarchy[fc]:
                    return list(hierarchy[fc][prob].keys())
            
            elif level == 'remedy' and len(parent_path) >= 3:
                fc, prob, cause = parent_path[0], parent_path[1], parent_path[2]
                if (fc in hierarchy and prob in hierarchy[fc] and 
                    cause in hierarchy[fc][prob]):
                    return hierarchy[fc][prob][cause]
        
        except (KeyError, IndexError):
            pass
        
        return []
    
    def _prepare_data_for_level(self, df: pd.DataFrame, level: str, parent_path: List[str] = None):
        """Prepare data for training a specific level - using combined text"""
        if parent_path:
            # Filter data based on parent path
            mask = pd.Series([True] * len(df))
            level_map = ['failure_class', 'problem', 'cause', 'remedy']
            
            for i, parent_val in enumerate(parent_path):
                mask &= (df[level_map[i]] == parent_val)
            
            filtered_df = df[mask].copy()
        else:
            filtered_df = df.copy()
        
        if len(filtered_df) == 0:
            return None, None, None
        
        # Prepare features and labels using combined text
        texts = filtered_df['combined_text'].astype(str).tolist()
        labels = filtered_df[level].tolist()
        
        return texts, labels, filtered_df
    
    def fit(self, df: pd.DataFrame):
        """Train hierarchical models with combined text"""
        self.logger.info("Starting enhanced hierarchical model training with combined text...")
        
        # Ensure combined text exists
        if 'combined_text' not in df.columns:
            raise ValueError("DataFrame must have 'combined_text' column. Use prepare_combined_text() first.")
        
        # Build training hierarchy first
        self._build_training_hierarchy(df)
        
        # Train models for each level
        levels = ['failure_class', 'problem', 'cause', 'remedy']
        
        for level_idx, level in enumerate(levels):
            self.logger.info(f"Training models for level: {level}")
            
            if level_idx == 0:
                # Root level - train on all data
                texts, labels, _ = self._prepare_data_for_level(df, level)
                
                if texts and labels:
                    # Encode labels
                    le = LabelEncoder()
                    encoded_labels = le.fit_transform(labels)
                    self.label_encoders[level] = le
                    
                    # Extract features from combined text
                    features = self.feature_engineer.extract_features(texts, fit=True)
                    
                    # Train model
                    model = self._create_model()
                    model.fit(features, encoded_labels)
                    self.models[level] = model
                    
                    # Evaluate
                    cv_scores = cross_val_score(model, features, encoded_labels, cv=min(5, len(set(labels))))
                    self.logger.info(f"{level} - CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            else:
                # Train models for each unique parent path
                parent_levels = levels[:level_idx]
                
                # Get all unique parent paths from training data
                parent_paths = df[parent_levels].drop_duplicates().values.tolist()
                
                level_models = {}
                level_encoders = {}
                
                for parent_path in parent_paths:
                    path_key = tuple(parent_path)
                    
                    texts, labels, filtered_df = self._prepare_data_for_level(df, level, parent_path)
                    
                    if texts and labels and len(set(labels)) > 1:  # Need at least 2 classes
                        # Encode labels for this path
                        le = LabelEncoder()
                        encoded_labels = le.fit_transform(labels)
                        level_encoders[path_key] = le
                        
                        # Extract features from combined text
                        features = self.feature_engineer.extract_features(texts, fit=False)
                        
                        # Train model for this path
                        model = self._create_model('rf')  # Use simpler model for smaller datasets
                        model.fit(features, encoded_labels)
                        level_models[path_key] = model
                        
                        self.logger.info(f"Trained {level} model for path {path_key} with {len(texts)} samples")
                
                self.models[level] = level_models
                self.label_encoders[level] = level_encoders
        
        self.logger.info("Enhanced hierarchical model training with combined text completed!")
    
    def predict_with_confidence_enhanced(self, description: str, remark: str = "") -> Dict:
        """Enhanced prediction with improved feedback integration"""
        # CHANGE: Ensure consistent text preprocessing
        desc_clean = str(description).strip() if description else ""
        remark_clean = str(remark).strip() if remark else ""
        
        combined_text = f"{desc_clean} | {remark_clean}" if remark_clean else desc_clean
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()  # Add this line
        if combined_text.endswith('|'):
            combined_text = combined_text[:-1].strip()
        
        self.logger.info(f"Processing prediction for: '{combined_text[:50]}...'")
        
        result = self._predict_with_combined_text(combined_text)
        result = self._apply_feedback_boost_to_single_result(combined_text, result)
        
        # CHANGE: Add alternative paths but keep in standard format
        try:
            multi_result = self.predict_multiple_paths(combined_text, "", top_k=3)
            # Store alternatives without changing main structure
            result['alternative_paths'] = multi_result['top_paths'][1:] if len(multi_result['top_paths']) > 1 else []
            result['debug_info'] = {
                'pattern_info': self.debug_pattern_matching(combined_text),
                'applied_feedback_boost': result.get('feedback_boost', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error getting alternative paths: {e}")
            result['alternative_paths'] = []
            result['debug_info'] = {'error': str(e)}
        
        # CHANGE: Always return standard format
        return result

    
    def _apply_feedback_boost_to_single_result(self, text: str, result: Dict) -> Dict:
        """Apply feedback boost to single result with restricted matching"""
        # Parse input text
        input_desc, input_remark = self._parse_combined_text(text)
        
        # Get the predicted path
        current_path = result.get('complete_path', [])
        if not current_path:
            return result
        
        path_tuple = tuple(current_path)
        
        # Find matching patterns with restricted criteria
        total_boost = 0.0
        matched_patterns = []
        
        self.logger.info(f"Checking restricted feedback for path: {path_tuple}")
        
        # CHANGE: Handle the correct pattern_weights structure
        for composite_key, weight in self.pattern_weights.items():
            (pattern_key, stored_path) = composite_key
            stored_desc, stored_remark = pattern_key
            # Get current path from result
            current_path = result.get('complete_path', [])
            path_tuple = tuple(current_path)

            if stored_path == path_tuple:
                # CHANGE: Extract description and remark from pattern_key tuple
                stored_desc, stored_remark = pattern_key
                
                # Check exact description match
                if stored_desc == input_desc.lower().strip():
                    # Check remark similarity
                    remark_match = True
                    if stored_remark and input_remark:
                        similarity = self._calculate_similarity(stored_remark, input_remark)
                        remark_match = similarity >= 0.4
                    elif not stored_remark and not input_remark:
                        remark_match = True
                    else:
                        remark_match = False
                    
                    if remark_match:
                        total_boost += weight
                        matched_patterns.append((f"exact_match:{stored_desc[:20]}...", weight))
                        self.logger.info(f"Found exact match with sufficient remark similarity: weight {weight}")
        
        if total_boost > 0:
            # Apply boost to overall confidence
            old_confidence = result.get('overall_confidence', 0.0)
            result['overall_confidence'] = min(1.0, old_confidence + total_boost)
            result['feedback_boost'] = total_boost
            
            # Update individual confidences proportionally
            boost_factor = result['overall_confidence'] / old_confidence if old_confidence > 0 else 1.0
            
            if 'confidences' in result:
                for level in result['confidences']:
                    result['confidences'][level] = min(1.0, result['confidences'][level] * boost_factor)
            
            self.logger.info(f"Applied restricted feedback boost: {old_confidence:.3f} -> {result['overall_confidence']:.3f}")
        else:
            result['feedback_boost'] = 0.0
            self.logger.info("No restricted feedback boost applied - no exact matches with sufficient similarity")
        
        return result
    def _predict_base_with_confidence(self, combined_text: str) -> Dict:
        """Base prediction method without feedback (renamed from original predict_with_confidence)"""
        result = {
            'predictions': {},
            'confidences': {},
            'complete_path': [],
            'overall_confidence': 1.0,
            'fallback_used': False,
            'source': {},
            'input_text': combined_text
        }
        
        levels = ['failure_class', 'problem', 'cause', 'remedy']
        current_path = []
        
        for level_idx, level in enumerate(levels):
            if level_idx == 0:
                # Root level prediction
                features = self.feature_engineer.extract_features([combined_text], fit=False)
                
                if level in self.models:
                    model = self.models[level]
                    probabilities = model.predict_proba(features)[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    predicted_label = self.label_encoders[level].inverse_transform([predicted_idx])[0]
                    
                    result['predictions'][level] = predicted_label
                    result['confidences'][level] = confidence
                    result['source'][level] = 'training_model'
                    current_path.append(predicted_label)
                    result['overall_confidence'] *= confidence
            
            else:
                # Child level prediction
                path_key = tuple(current_path)
                
                if (level in self.models and 
                    path_key in self.models[level] and 
                    path_key in self.label_encoders[level]):
                    
                    # Use trained model
                    model = self.models[level][path_key]
                    le = self.label_encoders[level][path_key]
                    
                    features = self.feature_engineer.extract_features([combined_text], fit=False)
                    probabilities = model.predict_proba(features)[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    predicted_label = le.inverse_transform([predicted_idx])[0]
                    
                    result['predictions'][level] = predicted_label
                    result['confidences'][level] = confidence
                    result['source'][level] = 'training_model'
                    current_path.append(predicted_label)
                    result['overall_confidence'] *= confidence
                    
                else:
                    # Use enhanced fallback
                    result['fallback_used'] = True
                    fallback_prediction = self._get_fallback_prediction(level, current_path)
                    
                    result['predictions'][level] = fallback_prediction
                    result['confidences'][level] = 0.6
                    result['source'][level] = 'hybrid_fallback'
                    current_path.append(fallback_prediction)
                    result['overall_confidence'] *= 0.6
        
        result['complete_path'] = current_path
        return result
    def _get_feedback_influenced_paths(self, text: str, top_k: int) -> List[Dict]:
        """Get paths that match feedback patterns"""
        text_lower = text.lower().strip()
        feedback_paths = []
        
        # Group patterns by path
        path_scores = {}
        for (pattern, path), weight in self.pattern_weights.items():
            if pattern.lower() in text_lower:
                path_tuple = tuple(path)
                if path_tuple not in path_scores:
                    path_scores[path_tuple] = 0.0
                path_scores[path_tuple] += weight
        
        # Convert to path results
        for path_tuple, total_score in path_scores.items():
            path_result = {
                'path': list(path_tuple),
                'overall_confidence': min(1.0, 0.5 + total_score),  # Base confidence + feedback
                'feedback_boost': total_score,
                'source': 'feedback_patterns',
                'matched_patterns': []
            }
            
            # Add individual confidences
            path_result['confidences'] = {
                level: path_result['overall_confidence'] 
                for level in ['failure_class', 'problem', 'cause', 'remedy']
            }
            
            feedback_paths.append(path_result)
        
        # Sort by confidence
        feedback_paths.sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        return feedback_paths[:top_k]
    
    def _get_model_predicted_paths(self, combined_text: str, top_k: int) -> List[Dict]:
        """Get paths from model predictions (existing logic)"""
        model_paths = []
        levels = ['failure_class', 'problem', 'cause', 'remedy']
        
        # Get top-k predictions for failure_class (existing logic)
        features = self.feature_engineer.extract_features([combined_text], fit=False)
        
        if 'failure_class' in self.models:
            model = self.models['failure_class']
            probabilities = model.predict_proba(features)[0]
            
            # Get top-k failure classes
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            for idx in top_indices:
                fc = self.label_encoders['failure_class'].inverse_transform([idx])[0]
                confidence = probabilities[idx]
                
                path_result = self._predict_complete_path_from_fc(combined_text, fc, confidence)
                path_result['source'] = 'model_prediction'
                path_result['feedback_boost'] = 0.0
                model_paths.append(path_result)
        
        return model_paths
    

    def _get_feedback_influenced_paths(self, text: str, top_k: int) -> List[Dict]:
        """Get paths that match restricted feedback patterns"""
        input_desc, input_remark = self._parse_combined_text(text)
        feedback_paths = []
        
        # Group patterns by path with restricted matching
        path_scores = {}
        # CHANGE: Handle the correct structure
        for composite_key, weight in self.pattern_weights.items():
            (pattern_key, path_key) = composite_key
            stored_desc, stored_remark = pattern_key
            
            # Check exact description match
            if stored_desc == input_desc.lower().strip():
                # Check remark similarity
                remark_match = True
                if stored_remark and input_remark:
                    similarity = self._calculate_similarity(stored_remark, input_remark)
                    remark_match = similarity >= 0.4
                elif not stored_remark and not input_remark:
                    remark_match = True
                else:
                    remark_match = False
                
                if remark_match:
                    if path_key not in path_scores:
                        path_scores[path_key] = 0.0
                    path_scores[path_key] += weight
        
        # Convert to path results
        for path_tuple, total_score in path_scores.items():
            path_result = {
                'path': list(path_tuple),
                'overall_confidence': min(1.0, 0.5 + total_score),
                'feedback_boost': total_score,
                'source': 'restricted_feedback_patterns',
                'matched_patterns': []
            }
            
            # Add individual confidences
            path_result['confidences'] = {
                level: path_result['overall_confidence'] 
                for level in ['failure_class', 'problem', 'cause', 'remedy']
            }
            
            feedback_paths.append(path_result)
        
        # Sort by confidence
        feedback_paths.sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        return feedback_paths[:top_k]
    
    def _combine_and_rank_paths(self, feedback_paths: List[Dict], model_paths: List[Dict], top_k: int) -> List[Dict]:
        """Combine feedback and model paths, removing duplicates and ranking"""
        all_paths = []
        seen_paths = set()
        
        # Add feedback paths first (higher priority)
        for path_result in feedback_paths:
            path_tuple = tuple(path_result['path'])
            if path_tuple not in seen_paths:
                all_paths.append(path_result)
                seen_paths.add(path_tuple)
        
        # Add model paths if not already present
        for path_result in model_paths:
            path_tuple = tuple(path_result['path'])
            if path_tuple not in seen_paths:
                all_paths.append(path_result)
                seen_paths.add(path_tuple)
        
        # Sort by overall confidence
        all_paths.sort(key=lambda x: x['overall_confidence'], reverse=True)
        
        return all_paths
    
    def _predict_with_combined_text(self, combined_text: str) -> Dict:
        """Internal method to predict with combined text"""
        result = {
            'predictions': {},
            'confidences': {},
            'complete_path': [],
            'overall_confidence': 1.0,
            'fallback_used': False,
            'source': {},
            'input_text': combined_text,
            'feedback_boost': 0.0  # CHANGE: Initialize feedback_boost
        }
        
        levels = ['failure_class', 'problem', 'cause', 'remedy']
        current_path = []
        
        for level_idx, level in enumerate(levels):
            if level_idx == 0:
                # Root level prediction
                features = self.feature_engineer.extract_features([combined_text], fit=False)
                
                if level in self.models:
                    model = self.models[level]
                    probabilities = model.predict_proba(features)[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    predicted_label = self.label_encoders[level].inverse_transform([predicted_idx])[0]
                    
                    result['predictions'][level] = predicted_label
                    result['confidences'][level] = confidence
                    result['source'][level] = 'training_model'
                    current_path.append(predicted_label)
                    result['overall_confidence'] *= confidence
            
            else:
                # Child level prediction
                path_key = tuple(current_path)
                
                if (level in self.models and 
                    path_key in self.models[level] and 
                    path_key in self.label_encoders[level]):
                    
                    # Use trained model
                    model = self.models[level][path_key]
                    le = self.label_encoders[level][path_key]
                    
                    features = self.feature_engineer.extract_features([combined_text], fit=False)
                    probabilities = model.predict_proba(features)[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    predicted_label = le.inverse_transform([predicted_idx])[0]
                    
                    result['predictions'][level] = predicted_label
                    result['confidences'][level] = confidence
                    result['source'][level] = 'training_model'
                    current_path.append(predicted_label)
                    result['overall_confidence'] *= confidence
                    
                else:
                    result['fallback_used'] = True
                    fallback_prediction = self._get_fallback_prediction(level, current_path, combined_text)
                    
                    result['predictions'][level] = fallback_prediction
                    result['confidences'][level] = 0.7  # INCREASE: from 0.6 to 0.7 for intelligent fallback
                    result['source'][level] = 'intelligent_fallback'  # CHANGE: Update source name
                    current_path.append(fallback_prediction)
                    result['overall_confidence'] *= 0.7  # Update multiplier
        
        result['complete_path'] = current_path
        return result

    def _rank_paths_with_concept_awareness(self, paths: List[Dict], text: str) -> List[Dict]:
        """ADD: Rank paths considering concept coherence"""
        
        text_lower = text.lower()
        
        # Calculate concept coherence for each path
        for path_result in paths:
            path = path_result['path']
            coherence_score = 0.0
            
            # Check if entire path makes conceptual sense
            for concept_name, concept_data in self.concept_mappings.items():
                # Count text matches to this concept
                text_matches = sum(1 for keyword in concept_data['keywords'] 
                                if keyword in text_lower)
                
                if text_matches > 0:
                    # Count path elements in this concept
                    path_matches = 0
                    for level, level_key in [('failure_class', 'failure_classes'), 
                                        ('problem', 'problems'), 
                                        ('cause', 'causes'), 
                                        ('remedy', 'remedies')]:
                        level_idx = ['failure_class', 'problem', 'cause', 'remedy'].index(level)
                        if level_idx < len(path):
                            level_options = concept_data.get(level_key, [])
                            if path[level_idx] in level_options:
                                path_matches += 1
                    
                    # Coherence = text_relevance * path_consistency
                    if len(concept_data['keywords']) > 0:
                        concept_coherence = (text_matches / len(concept_data['keywords'])) * (path_matches / 4)
                        coherence_score = max(coherence_score, concept_coherence)
            
            # Apply coherence boost
            path_result['concept_coherence'] = coherence_score
            path_result['overall_confidence'] = min(1.0, 
                path_result['overall_confidence'] + coherence_score * 0.2)
        
        # Re-sort by updated confidence
        paths.sort(key=lambda x: x['overall_confidence'], reverse=True)
        return paths
    def predict_with_confidence(self, description: str, remark: str = "") -> Dict:
        """Predict complete path with confidence scores using combined text"""
        ### CHANGE 1: Use enhanced prediction method that applies feedback ###
        return self.predict_with_confidence_enhanced(description, remark)
    def _get_fallback_prediction(self, level: str, parent_path: List[str], text: str = "") -> str:
        """REPLACE: Enhanced fallback with semantic intelligence"""
        return self._get_intelligent_fallback(level, parent_path, text)
    
    def _path_matches(self, full_path: tuple, partial_path: list, level: str) -> bool:
        """Check if a full path matches partial path up to given level"""
        level_map = {
            'failure_class': 0,
            'problem': 1, 
            'cause': 2,
            'remedy': 3
        }
        
        level_idx = level_map.get(level, 0)
        
        for i in range(min(len(partial_path), level_idx + 1)):
            if i < len(full_path) and full_path[i] != partial_path[i]:
                return False
        
        return True
    
    def get_hierarchy_statistics(self) -> Dict:
        """Get statistics about the hierarchies - FIXED VERSION"""
        stats = {}
        
        # Training hierarchy stats
        if self.training_hierarchy:
            total_paths = len(self.path_frequencies)  # FIXED: removed sum()
            avg_frequency = sum(self.path_frequencies.values()) / len(self.path_frequencies) if self.path_frequencies else 0
            
            stats['training_hierarchy'] = {
                'failure_classes': len(self.training_hierarchy),
                'total_paths': total_paths,
                'avg_frequency': avg_frequency
            }
        
        # Full hierarchy stats
        if self.full_hierarchy:
            total_paths = 0
            for fc in self.full_hierarchy:
                for prob in self.full_hierarchy[fc]:
                    for cause in self.full_hierarchy[fc][prob]:
                        total_paths += len(self.full_hierarchy[fc][prob][cause])
            
            stats['full_hierarchy'] = {
                'failure_classes': len(self.full_hierarchy),
                'total_paths': total_paths
            }
        
        return stats
        
    def save_model(self, filepath: str):
        """Save enhanced model"""
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_engineer': self.feature_engineer,
            'training_hierarchy': self.training_hierarchy,
            'full_hierarchy': self.full_hierarchy,
            'path_frequencies': self.path_frequencies,
            'confidence_threshold': self.confidence_threshold,
            'use_ensemble': self.use_ensemble
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Enhanced model saved to {filepath}")
    def predict_multiple_paths(self, description: str, remark: str = "", top_k: int = 3) -> Dict:
        """Predict multiple possible paths with feedback integration"""
        combined_text = f"{description} | {remark}".strip()
        if combined_text.endswith('|'):
            combined_text = combined_text[:-1].strip()
        
        result = {
            'input_text': combined_text,
            'top_paths': [],
            'reasoning': {}
        }
        
        # CHANGE: Get feedback-influenced paths first
        feedback_paths = self._get_feedback_influenced_paths(combined_text, top_k)
        
        # CHANGE: Get model-predicted paths
        model_paths = self._get_model_predicted_paths(combined_text, top_k)
        
        # CHANGE: Combine and rank all paths - FIX: Define all_paths first
        all_paths = self._combine_and_rank_paths(feedback_paths, model_paths, top_k)
        all_paths = self._rank_paths_with_concept_awareness(all_paths, combined_text)
        
        result['top_paths'] = all_paths[:top_k]
        return result
    def _predict_complete_path_from_fc(self, combined_text: str, failure_class: str, fc_confidence: float) -> Dict:
        """Predict complete path starting from a given failure class"""
        path = [failure_class]
        confidences = [fc_confidence]
        sources = ['training_model']
        overall_confidence = fc_confidence
        
        levels = ['problem', 'cause', 'remedy']
        
        for level in levels:
            path_key = tuple(path)
            
            if (level in self.models and 
                path_key in self.models[level] and 
                path_key in self.label_encoders[level]):
                
                # Use trained model
                model = self.models[level][path_key]
                le = self.label_encoders[level][path_key]
                
                features = self.feature_engineer.extract_features([combined_text], fit=False)
                probabilities = model.predict_proba(features)[0]
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                
                predicted_label = le.inverse_transform([predicted_idx])[0]
                
                path.append(predicted_label)
                confidences.append(confidence)
                sources.append('training_model')
                overall_confidence *= confidence
            else:
                # Use fallback
                fallback_prediction = self._get_fallback_prediction(level, path)
                path.append(fallback_prediction)
                confidences.append(0.6)
                sources.append('hybrid_fallback')
                overall_confidence *= 0.6
        
        return {
            'path': path,
            'confidences': dict(zip(['failure_class', 'problem', 'cause', 'remedy'], confidences)),
            'overall_confidence': overall_confidence,
            'sources': dict(zip(['failure_class', 'problem', 'cause', 'remedy'], sources))
        }
    # Add this method to help debug pattern matching
    def debug_pattern_matching(self, text: str) -> Dict:
        """Debug method to show pattern matching details"""
        text_lower = text.lower().strip()
        
        debug_info = {
            'input_text': text,
            'processed_text': text_lower,
            'total_patterns': len(self.pattern_weights),
            'matching_patterns': [],
            'potential_paths': {}
        }
        
        # Find all matching patterns
        for (pattern, path), weight in self.pattern_weights.items():
            if pattern.lower() in text_lower:
                debug_info['matching_patterns'].append({
                    'pattern': pattern,
                    'path': path,
                    'weight': weight
                })
                
                if path not in debug_info['potential_paths']:
                    debug_info['potential_paths'][path] = 0.0
                debug_info['potential_paths'][path] += weight
        
        return debug_info


    def get_feedback_stats(self) -> Dict:
        """Get statistics about stored feedback"""
        if not self.feedback_data:
            return {"feedback_count": 0, "patterns": 0}
        
        pattern_count = len(self.pattern_weights)
        recent_feedback = self.feedback_data[-5:] if len(self.feedback_data) >= 5 else self.feedback_data
        
        return {
            "feedback_count": len(self.feedback_data),
            "pattern_weights": pattern_count,
            "recent_feedback": [
                {
                    "text": fb["text"][:50] + "..." if len(fb["text"]) > 50 else fb["text"],
                    "path": fb["correct_path"],
                    "timestamp": fb["timestamp"]
                }
                for fb in recent_feedback
            ],
            "sample_patterns": list(self.pattern_weights.keys())[:10]  # Show first 10 patterns
        }

    def load_model(self, filepath: str):
        """Load enhanced model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.feature_engineer = model_data['feature_engineer']
        self.training_hierarchy = model_data.get('training_hierarchy', {})
        self.full_hierarchy = model_data.get('full_hierarchy', {})
        self.path_frequencies = model_data.get('path_frequencies', {})
        self.confidence_threshold = model_data['confidence_threshold']
        self.use_ensemble = model_data.get('use_ensemble', True)
        
        self.logger.info(f"Enhanced model loaded from {filepath}")
