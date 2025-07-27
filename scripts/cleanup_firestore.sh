#!/bin/bash

# Firestore Collections Cleanup Script
# Deletes all documents from Athena's Firestore collections

echo "🧹 Cleaning up Firestore collections..."

# List of collections to clean up
collections=(
    "agent_memories"
    "agent_state" 
    "pool_profiles"
    "pool_metrics"
    "pattern_correlations"
    "observation_metrics"
    "observed_patterns"
    "pattern_confidence"
    "gas_prices"
    "market_data"
    "agent_decisions"
    "agent_emotions"
    "strategy_outcomes"
    "error_logs"
    "risk_assessments"
    "circuit_breaker_events"
    "checkpoints"
    "performance_metrics"
)

# Get current project
PROJECT_ID=$(gcloud config get-value project)
echo "Project: $PROJECT_ID"

# Function to delete all documents in a collection
cleanup_collection() {
    local collection=$1
    echo "Cleaning collection: $collection"
    
    # Delete all documents in the collection
    gcloud firestore collection-groups delete "$collection" --project="$PROJECT_ID" --quiet 2>/dev/null || {
        echo "Collection $collection doesn't exist or already empty"
    }
}

# Clean each collection
for collection in "${collections[@]}"; do
    cleanup_collection "$collection"
done

echo "✅ Firestore cleanup completed!"