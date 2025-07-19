"""
Firestore client for persistent storage
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from decimal import Decimal

logger = logging.getLogger(__name__)


class FirestoreClient:
    """Client for interacting with Firestore."""
    
    def __init__(self, project_id: str):
        """Initialize Firestore client."""
        self.db = firestore.Client(project=project_id)
        logger.info(f"Firestore client initialized for project: {project_id}")
        
    def save_agent_state(self, state: Dict[str, Any]) -> None:
        """Save current agent state."""
        try:
            doc_ref = self.db.collection('agent_state').document('current')
            
            # Convert Decimal to float for Firestore
            clean_state = self._clean_for_firestore(state)
            clean_state['last_update'] = datetime.utcnow()
            
            doc_ref.set(clean_state)
            logger.info("Agent state saved to Firestore")
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            
    def save_cycle_result(self, cycle_number: int, result: Dict[str, Any]) -> None:
        """Save reasoning cycle result."""
        try:
            doc_ref = self.db.collection('cycles').document(f'cycle_{cycle_number}')
            
            clean_result = self._clean_for_firestore(result)
            clean_result['cycle_number'] = cycle_number
            clean_result['timestamp'] = datetime.utcnow()
            
            doc_ref.set(clean_result)
            logger.info(f"Cycle {cycle_number} result saved")
        except Exception as e:
            logger.error(f"Failed to save cycle result: {e}")
            
    def save_position(self, position: Dict[str, Any]) -> str:
        """Save a new position."""
        try:
            clean_position = self._clean_for_firestore(position)
            clean_position['created_at'] = datetime.utcnow()
            clean_position['status'] = 'active'
            
            doc_ref = self.db.collection('positions').add(clean_position)[1]
            logger.info(f"Position saved with ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            return ""
            
    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics."""
        try:
            doc_ref = self.db.collection('performance').document('summary')
            
            clean_metrics = self._clean_for_firestore(metrics)
            clean_metrics['last_update'] = datetime.utcnow()
            
            doc_ref.set(clean_metrics, merge=True)
            logger.info("Performance metrics updated")
        except Exception as e:
            logger.error(f"Failed to update performance: {e}")
            
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions."""
        try:
            positions = []
            docs = self.db.collection('positions').where(
                filter=FieldFilter('status', '==', 'active')
            ).stream()
            
            for doc in docs:
                position = doc.to_dict()
                position['id'] = doc.id
                positions.append(position)
                
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
            
    def _clean_for_firestore(self, data: Any) -> Any:
        """Clean data for Firestore storage."""
        if isinstance(data, dict):
            return {k: self._clean_for_firestore(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_firestore(item) for item in data]
        elif isinstance(data, Decimal):
            return float(data)
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return data
            
    def save_pattern(self, pattern: Dict[str, Any]) -> str:
        """Save a discovered pattern during observation."""
        try:
            clean_pattern = self._clean_for_firestore(pattern)
            clean_pattern['discovered_at'] = datetime.utcnow()
            
            doc_ref = self.db.collection('observed_patterns').add(clean_pattern)[1]
            logger.info(f"Pattern saved with ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
            return ""
            
    def update_pattern_confidence(self, pattern_id: str, confidence: float, success: bool) -> None:
        """Update pattern confidence based on outcomes."""
        try:
            doc_ref = self.db.collection('pattern_confidence').document(pattern_id)
            
            # Get existing data or create new
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                occurrences = data.get('occurrences', 0) + 1
                successes = data.get('successes', 0) + (1 if success else 0)
            else:
                occurrences = 1
                successes = 1 if success else 0
            
            # Update confidence
            new_confidence = successes / occurrences
            
            doc_ref.set({
                'pattern_id': pattern_id,
                'confidence': new_confidence,
                'occurrences': occurrences,
                'successes': successes,
                'last_update': datetime.utcnow()
            })
            
            logger.info(f"Pattern {pattern_id} confidence updated to {new_confidence:.2f}")
        except Exception as e:
            logger.error(f"Failed to update pattern confidence: {e}")
            
    def save_observation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save observation period metrics."""
        try:
            doc_ref = self.db.collection('observation_metrics').document('current')
            
            clean_metrics = self._clean_for_firestore(metrics)
            clean_metrics['last_update'] = datetime.utcnow()
            
            doc_ref.set(clean_metrics, merge=True)
            logger.info("Observation metrics saved")
        except Exception as e:
            logger.error(f"Failed to save observation metrics: {e}")
            
    def get_high_confidence_patterns(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Get patterns with high confidence scores."""
        try:
            patterns = []
            
            # Get pattern confidence scores
            confidence_docs = self.db.collection('pattern_confidence').where(
                filter=FieldFilter('confidence', '>=', min_confidence)
            ).stream()
            
            pattern_ids = []
            confidence_map = {}
            
            for doc in confidence_docs:
                data = doc.to_dict()
                pattern_ids.append(data['pattern_id'])
                confidence_map[data['pattern_id']] = data['confidence']
            
            # Get actual patterns
            if pattern_ids:
                for pattern_id in pattern_ids:
                    pattern_doc = self.db.collection('observed_patterns').document(pattern_id).get()
                    if pattern_doc.exists:
                        pattern = pattern_doc.to_dict()
                        pattern['id'] = pattern_id
                        pattern['confidence'] = confidence_map[pattern_id]
                        patterns.append(pattern)
                        
            return patterns
        except Exception as e:
            logger.error(f"Failed to get high confidence patterns: {e}")
            return []