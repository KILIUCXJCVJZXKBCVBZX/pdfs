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
            'failure_classes': ['PLUMBING', 'Pipe', 'Water Supply', 'Water Heater','toilet'],  # ADD Water Heater
            'related_concepts': ['electrical_heating', 'water_systems'],  # ADD relationships
            'problems': ['Water heater not working', 'Leakage', 'Water Leak', 'No Hot Water'],
            'causes': ['Water heater defected', 'Leakage In System', 'Pipe Damage'],
            'remedies': ['Replace', 'Repair/Replace', 'Fix Leak']
        },
        'electrical_concepts': {
            'keywords': ['power', 'electrical', 'circuit', 'breaker', 'wiring', 'voltage',
                        'switch', 'outlet', 'lighting', 'heater'],
            'failure_classes': ['CIRCUIT BREAKER', 'Heater', 'Lighting Systems', 'LGTSYS'],
            'related_concepts': ['plumbing_concepts'],  # ADD: electrical heaters relate to plumbing
            'problems': ['Not Working', 'Unable To Reset', 'Deformation'],
            'causes': ['No Power', 'Overloaded', 'Loose Connections'],
            'remedies': ['Check Incoming Feeder', 'Check Load Side', 'Replace & tighten']
        },
        # ADD: New water systems concept
        'water_systems': {
            'keywords': ['water', 'heating', 'temperature', 'hot', 'cold'],
            'failure_classes': ['Water Heater', 'PLUMBING', 'Heater'],
            'related_concepts': ['plumbing_concepts', 'electrical_concepts']
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
        """Enhanced fallback with path similarity and concept relationships"""
        
        # STEP 1: Try to find similar paths with same failure_class + problem
        if level in ['cause', 'remedy'] and len(parent_path) >= 2:
            similar_paths = self._find_similar_paths_same_fc_problem(parent_path[:2])
            if similar_paths:
                # Score by text relevance
                best_option = self._score_options_by_text_relevance(
                    similar_paths, level, text, parent_path
                )
                if best_option:
                    return best_option
        
        # STEP 2: Try related concepts fallback
        related_options = self._get_related_concept_options(level, parent_path, text)
        if related_options:
            return self._score_options_by_text_relevance(
                related_options, level, text, parent_path
            )
        
        # STEP 3: Original logic as final fallback
        valid_options = self._get_valid_options(level, parent_path)
        if valid_options:
            return self._score_options_by_frequency(valid_options, parent_path)
        
        return self._get_basic_fallback(level)
    def _get_related_concept_options(self, level: str, parent_path: List[str], text: str) -> List[str]:
        """Get options from related concepts"""
        if not parent_path:
            return []
        
        current_fc = parent_path[0]
        related_options = []
        
        # Find which concept the current failure_class belongs to
        current_concept = None
        for concept_name, concept_data in self.concept_mappings.items():
            if current_fc in concept_data.get('failure_classes', []):
                current_concept = concept_name
                break
        
        if not current_concept:
            return []
        
        # Get related concepts
        related_concepts = self.concept_mappings[current_concept].get('related_concepts', [])
        
        for related_name in related_concepts:
            if related_name in self.concept_mappings:
                related_data = self.concept_mappings[related_name]
                
                # Check if text matches this related concept
                text_lower = text.lower()
                matches = sum(1 for keyword in related_data['keywords'] if keyword in text_lower)
                
                if matches > 0:  # Text is relevant to this related concept
                    level_key = f'{level}s' if level != 'failure_class' else 'failure_classes'
                    if level == 'cause':
                        level_key = 'causes'
                    elif level == 'remedy':
                        level_key = 'remedies'
                    
                    related_options.extend(related_data.get(level_key, []))
        
        return list(set(related_options))
    def _score_options_by_text_relevance(self, options: List[str], level: str, text: str, parent_path: List[str]) -> str:
        """Score options based on text relevance and return best match"""
        if not options:
            return None
        
        text_lower = text.lower()
        option_scores = {}
        
        for option in options:
            score = 0.0
            option_lower = option.lower()
            
            # Direct keyword matching
            if option_lower in text_lower:
                score += 1.0
            
            # Partial matching
            option_words = option_lower.split()
            for word in option_words:
                if word in text_lower:
                    score += 0.5
            
            # Concept coherence scoring
            for concept_name, concept_data in self.concept_mappings.items():
                if option in concept_data.get(f'{level}s', concept_data.get('failure_classes', [])):
                    # Count text matches to this concept
                    concept_matches = sum(1 for keyword in concept_data['keywords'] if keyword in text_lower)
                    if concept_matches > 0:
                        score += concept_matches * 0.3
            
            # Frequency boost from training data
            for path, freq in self.path_frequencies.items():
                if option in path:
                    score += min(freq * 0.01, 0.2)  # Cap frequency boost
            
            option_scores[option] = score
        
        # Return option with highest score
        return max(option_scores.items(), key=lambda x: x[1])[0]
    
    def _find_similar_paths_same_fc_problem(self, fc_problem_path: List[str]) -> List[str]:
        """Find all paths that share the same failure_class and problem"""
        if len(fc_problem_path) < 2:
            return []
        
        fc, problem = fc_problem_path[0], fc_problem_path[1]
        similar_options = []
        
        # Search in training hierarchy
        for path, freq in self.path_frequencies.items():
            if len(path) >= 2 and path[0] == fc and path[1] == problem:
                similar_options.extend(path[2:])  # Add causes and remedies
        
        # Search in full hierarchy
        if fc in self.full_hierarchy and problem in self.full_hierarchy[fc]:
            for cause in self.full_hierarchy[fc][problem]:
                similar_options.append(cause)
                similar_options.extend(self.full_hierarchy[fc][problem][cause])
        
        return list(set(similar_options))

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
            'hot water', 'cold water', 'pressure', 'flow','toilet'
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
        """Enhanced model prediction with concept awareness"""
        model_paths = []
        
        # Original model predictions
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
                model_paths.append(path_result)
        
        # ADD: Include related concept paths if confidence is low
        if model_paths and model_paths[0]['overall_confidence'] < 0.6:
            concept_paths = self._get_concept_aware_paths(combined_text, top_k // 2)
            model_paths.extend(concept_paths)
        
        return model_paths[:top_k]
    
    def _build_concept_coherent_path(self, text: str, failure_class: str, concept_data: dict) -> Dict:
        """Build a coherent path from concept data and failure class"""
        
        # Initialize path with the given failure class
        path = [failure_class]
        confidences = {'failure_class': 0.8}  # Base confidence for concept-based FC
        
        # Text analysis for relevance scoring
        text_lower = text.lower()
        
        # Step 1: Find the best problem for this concept and text
        problems = concept_data.get('problems', [])
        if problems:
            # Score problems based on text relevance
            problem_scores = {}
            for problem in problems:
                score = 0.0
                problem_lower = problem.lower()
                
                # Direct text match
                if problem_lower in text_lower:
                    score += 1.0
                
                # Word overlap scoring
                problem_words = problem_lower.split()
                text_words = text_lower.split()
                overlap = len(set(problem_words) & set(text_words))
                if len(problem_words) > 0:
                    score += overlap / len(problem_words) * 0.8
                
                # Concept keyword relevance
                concept_keywords = concept_data.get('keywords', [])
                keyword_matches = sum(1 for keyword in concept_keywords if keyword in problem_lower)
                if len(concept_keywords) > 0:
                    score += keyword_matches / len(concept_keywords) * 0.6
                
                problem_scores[problem] = score
            
            # Select best problem
            best_problem = max(problem_scores, key=problem_scores.get)
            path.append(best_problem)
            confidences['problem'] = min(0.9, 0.5 + problem_scores[best_problem])
        else:
            # Fallback to hierarchy-based problem
            hierarchy_problems = self._get_valid_options('problem', [failure_class])
            if hierarchy_problems:
                # Score hierarchy problems by text relevance
                best_problem = self._score_options_by_text_relevance(
                    hierarchy_problems, 'problem', text, [failure_class]
                )
                path.append(best_problem)
                confidences['problem'] = 0.6
            else:
                path.append('GENERAL PROBLEM')
                confidences['problem'] = 0.4
        
        # Step 2: Find the best cause for this concept and text
        causes = concept_data.get('causes', [])
        if causes:
            # Score causes based on text relevance and concept coherence
            cause_scores = {}
            for cause in causes:
                score = 0.0
                cause_lower = cause.lower()
                
                # Direct text match
                if cause_lower in text_lower:
                    score += 1.0
                
                # Word overlap scoring
                cause_words = cause_lower.split()
                text_words = text_lower.split()
                overlap = len(set(cause_words) & set(text_words))
                if len(cause_words) > 0:
                    score += overlap / len(cause_words) * 0.8
                
                # Check for related concept keywords
                concept_keywords = concept_data.get('keywords', [])
                for keyword in concept_keywords:
                    if keyword in cause_lower:
                        score += 0.3
                
                cause_scores[cause] = score
            
            # Select best cause
            best_cause = max(cause_scores, key=cause_scores.get)
            path.append(best_cause)
            confidences['cause'] = min(0.9, 0.5 + cause_scores[best_cause])
        else:
            # Fallback to hierarchy-based cause
            hierarchy_causes = self._get_valid_options('cause', path[:2])
            if hierarchy_causes:
                best_cause = self._score_options_by_text_relevance(
                    hierarchy_causes, 'cause', text, path[:2]
                )
                path.append(best_cause)
                confidences['cause'] = 0.6
            else:
                path.append('UNKNOWN CAUSE')
                confidences['cause'] = 0.4
        
        # Step 3: Find the best remedy for this concept
        remedies = concept_data.get('remedies', [])
        if remedies:
            # Score remedies based on problem-cause context and text
            remedy_scores = {}
            for remedy in remedies:
                score = 0.0
                remedy_lower = remedy.lower()
                
                # Direct text match
                if remedy_lower in text_lower:
                    score += 0.8
                
                # Context-based scoring (remedy should match problem/cause severity)
                problem_lower = path[1].lower() if len(path) > 1 else ""
                cause_lower = path[2].lower() if len(path) > 2 else ""
                
                # Simple heuristics for remedy appropriateness
                if 'replace' in remedy_lower:
                    if any(word in problem_lower + cause_lower for word in ['broken', 'defected', 'failed']):
                        score += 0.5
                elif 'repair' in remedy_lower:
                    if any(word in problem_lower + cause_lower for word in ['damage', 'leak', 'loose']):
                        score += 0.5
                elif 'check' in remedy_lower:
                    if any(word in problem_lower + cause_lower for word in ['no power', 'connection']):
                        score += 0.5
                
                # Concept keyword relevance
                concept_keywords = concept_data.get('keywords', [])
                keyword_matches = sum(1 for keyword in concept_keywords if keyword in remedy_lower)
                if len(concept_keywords) > 0:
                    score += keyword_matches / len(concept_keywords) * 0.4
                
                remedy_scores[remedy] = score
            
            # Select best remedy
            best_remedy = max(remedy_scores, key=remedy_scores.get)
            path.append(best_remedy)
            confidences['remedy'] = min(0.9, 0.5 + remedy_scores[best_remedy])
        else:
            # Fallback to hierarchy-based remedy
            hierarchy_remedies = self._get_valid_options('remedy', path[:3])
            if hierarchy_remedies:
                best_remedy = self._score_options_by_text_relevance(
                    hierarchy_remedies, 'remedy', text, path[:3]
                )
                path.append(best_remedy)
                confidences['remedy'] = 0.6
            else:
                # Generate contextual remedy
                if len(path) >= 3:
                    default_remedies = self._get_default_remedies(path[0], path[1], path[2])
                    path.append(default_remedies[0] if default_remedies else 'REPAIR')
                else:
                    path.append('REPAIR')
                confidences['remedy'] = 0.4
        
        # Calculate overall confidence
        overall_confidence = 1.0
        for level_conf in confidences.values():
            overall_confidence *= level_conf
        
        # Add concept coherence boost
        concept_keywords = concept_data.get('keywords', [])
        text_concept_matches = sum(1 for keyword in concept_keywords if keyword in text_lower)
        if len(concept_keywords) > 0:
            concept_relevance = text_concept_matches / len(concept_keywords)
            overall_confidence = min(1.0, overall_confidence + concept_relevance * 0.2)
        
        # Build result
        result = {
            'path': path,
            'overall_confidence': overall_confidence,
            'confidences': confidences,
            'source': 'concept_coherent_path',
            'concept_used': concept_data.get('concept_name', 'unknown'),
            'concept_relevance': concept_relevance if 'concept_relevance' in locals() else 0.0,
            'fallback_used': len(path) < 4  # True if we couldn't complete the full path
        }

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
    

    
    def _score_options_by_frequency(self, options: List[str], parent_path: List[str]) -> str:
        """Score options based on frequency in training data and return best match"""
        if not options:
            return None
        
        option_scores = {}
        
        for option in options:
            score = 0.0
            
            # Score based on frequency in path_frequencies
            for path, freq in self.path_frequencies.items():
                if option in path:
                    # Give higher score if option appears in a path that shares more context
                    shared_context = 0
                    for i, parent_item in enumerate(parent_path):
                        if i < len(path) and path[i] == parent_item:
                            shared_context += 1
                        else:
                            break
                    
                    # Weight frequency by shared context
                    context_weight = (shared_context + 1) / (len(parent_path) + 1)
                    score += freq * context_weight
            
            # Fallback: if no frequency data, give base score
            if score == 0.0:
                score = 1.0
            
            option_scores[option] = score
        
        # Return option with highest score
        if option_scores:
            return max(option_scores.items(), key=lambda x: x[1])[0]
        else:
            return options[0]  # Fallback to first option
    def _get_concept_aware_paths(self, text: str, max_paths: int) -> List[Dict]:
        """Generate paths based on concept analysis"""
        text_lower = text.lower()
        concept_paths = []
        
        # Analyze which concepts the text relates to
        concept_scores = {}
        for concept_name, concept_data in self.concept_mappings.items():
            score = sum(1 for keyword in concept_data['keywords'] if keyword in text_lower)
            if score > 0:
                concept_scores[concept_name] = score
        
        # Generate paths from top concepts
        for concept_name in sorted(concept_scores.keys(), key=lambda x: concept_scores[x], reverse=True):
            if len(concept_paths) >= max_paths:
                break
                
            concept_data = self.concept_mappings[concept_name]
            
            # Try to build a coherent path from this concept
            for fc in concept_data.get('failure_classes', []):
                if fc in self.training_hierarchy or fc in self.full_hierarchy:
                    path_result = self._build_concept_coherent_path(text, fc, concept_data)
                    if path_result:
                        concept_paths.append(path_result)
                        if len(concept_paths) >= max_paths:
                            break
        
        return concept_paths
    
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

    
    def _analyze_concept_patterns(self, df: pd.DataFrame):
        """Analyze patterns in training data to enhance concept relationships"""
        
        # Build co-occurrence matrix for failure classes
        fc_cooccurrence = {}
        
        for _, row in df.iterrows():
            text = row['combined_text'].lower()
            fc = row['failure_class']
            
            # Find which concepts this text and FC relate to
            for concept_name, concept_data in self.concept_mappings.items():
                text_matches = sum(1 for keyword in concept_data['keywords'] if keyword in text)
                
                if text_matches > 0 and fc in concept_data.get('failure_classes', []):
                    # This confirms the concept relationship
                    if concept_name not in fc_cooccurrence:
                        fc_cooccurrence[concept_name] = {}
                    fc_cooccurrence[concept_name][fc] = fc_cooccurrence[concept_name].get(fc, 0) + 1
        
        # Store for use in fallback
        self.concept_fc_patterns = fc_cooccurrence
