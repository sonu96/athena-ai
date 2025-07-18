#!/bin/bash

# Google Cloud Project Cleanup Script
# This script helps manage and clean up GCP projects

set -e

echo "🧹 Google Cloud Project Cleanup"
echo "=============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get current account
CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null)
echo -e "\nCurrent account: ${GREEN}${CURRENT_ACCOUNT}${NC}"

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
echo -e "Active project: ${BLUE}${CURRENT_PROJECT}${NC}"

# Function to calculate project age
get_project_age() {
    local create_time=$1
    local created_date=$(date -d "$create_time" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$create_time" +%s 2>/dev/null)
    local current_date=$(date +%s)
    local age_days=$(( (current_date - created_date) / 86400 ))
    echo $age_days
}

# List all projects with details
echo -e "\n📋 Your Google Cloud Projects:"
echo "------------------------------"

# Get projects in a more detailed format
projects_json=$(gcloud projects list --format=json)

# Parse and display projects
echo "$projects_json" | jq -r '.[] | "\(.projectId)|\(.name)|\(.createTime)|\(.lifecycleState)"' | while IFS='|' read -r id name created state; do
    age=$(get_project_age "$created")
    
    # Color code by age
    if [ $age -gt 90 ]; then
        age_color=$RED
    elif [ $age -gt 30 ]; then
        age_color=$YELLOW
    else
        age_color=$GREEN
    fi
    
    # Mark current project
    if [ "$id" == "$CURRENT_PROJECT" ]; then
        current_marker=" ${BLUE}[ACTIVE]${NC}"
    else
        current_marker=""
    fi
    
    printf "%-30s %-25s ${age_color}%3d days old${NC} %s%s\n" "$id" "$name" "$age" "$state" "$current_marker"
done

# Show project count
TOTAL_PROJECTS=$(echo "$projects_json" | jq '. | length')
echo -e "\nTotal projects: ${BLUE}$TOTAL_PROJECTS${NC}"

# Interactive cleanup options
echo -e "\n🔧 Cleanup Options:"
echo "1. Delete specific projects"
echo "2. Delete all projects except current"
echo "3. Delete projects older than N days"
echo "4. List project resources (to check before deletion)"
echo "5. Exit"

read -p "Choose an option (1-5): " OPTION

case $OPTION in
    1)
        # Delete specific projects
        echo -e "\nEnter project IDs to delete (comma-separated):"
        read -p "> " PROJECTS_TO_DELETE
        
        IFS=',' read -ra PROJECT_ARRAY <<< "$PROJECTS_TO_DELETE"
        for project in "${PROJECT_ARRAY[@]}"; do
            project=$(echo "$project" | xargs)
            
            if [ "$project" == "$CURRENT_PROJECT" ]; then
                echo -e "${YELLOW}⚠ Skipping active project: $project${NC}"
                continue
            fi
            
            echo -e "\n${RED}Delete project: $project?${NC}"
            read -p "Confirm (y/N): " CONFIRM
            
            if [[ $CONFIRM =~ ^[Yy]$ ]]; then
                if gcloud projects delete "$project" --quiet; then
                    echo -e "${GREEN}✓ Deleted: $project${NC}"
                else
                    echo -e "${RED}✗ Failed to delete: $project${NC}"
                fi
            fi
        done
        ;;
        
    2)
        # Delete all except current
        echo -e "\n${RED}⚠ WARNING: This will delete ALL projects except the current one!${NC}"
        echo -e "Current project: ${BLUE}$CURRENT_PROJECT${NC}"
        read -p "Are you SURE? Type 'DELETE ALL' to confirm: " CONFIRM
        
        if [ "$CONFIRM" == "DELETE ALL" ]; then
            echo "$projects_json" | jq -r '.[].projectId' | while read -r project; do
                if [ "$project" != "$CURRENT_PROJECT" ]; then
                    echo -e "\nDeleting: $project"
                    if gcloud projects delete "$project" --quiet; then
                        echo -e "${GREEN}✓ Deleted: $project${NC}"
                    else
                        echo -e "${RED}✗ Failed: $project${NC}"
                    fi
                fi
            done
        else
            echo "Cancelled."
        fi
        ;;
        
    3)
        # Delete old projects
        read -p "Delete projects older than how many days? " DAYS
        
        if ! [[ "$DAYS" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Invalid number${NC}"
            exit 1
        fi
        
        echo -e "\nProjects older than $DAYS days:"
        old_projects=()
        
        echo "$projects_json" | jq -r '.[] | "\(.projectId)|\(.createTime)"' | while IFS='|' read -r id created; do
            age=$(get_project_age "$created")
            if [ $age -gt $DAYS ] && [ "$id" != "$CURRENT_PROJECT" ]; then
                echo "- $id (${age} days old)"
                old_projects+=("$id")
            fi
        done
        
        if [ ${#old_projects[@]} -eq 0 ]; then
            echo "No projects found older than $DAYS days"
        else
            read -p "Delete these projects? (y/N): " CONFIRM
            if [[ $CONFIRM =~ ^[Yy]$ ]]; then
                for project in "${old_projects[@]}"; do
                    echo -e "\nDeleting: $project"
                    if gcloud projects delete "$project" --quiet; then
                        echo -e "${GREEN}✓ Deleted: $project${NC}"
                    else
                        echo -e "${RED}✗ Failed: $project${NC}"
                    fi
                done
            fi
        fi
        ;;
        
    4)
        # List project resources
        read -p "Enter project ID to inspect: " PROJECT_ID
        
        echo -e "\n📊 Resources in project: ${BLUE}$PROJECT_ID${NC}"
        echo "----------------------------------------"
        
        # Temporarily switch to the project
        ORIGINAL_PROJECT=$CURRENT_PROJECT
        gcloud config set project "$PROJECT_ID" --quiet
        
        # Check various resources
        echo -e "\n${YELLOW}Cloud Run services:${NC}"
        gcloud run services list 2>/dev/null || echo "None or no access"
        
        echo -e "\n${YELLOW}Firestore databases:${NC}"
        gcloud firestore databases list 2>/dev/null || echo "None or no access"
        
        echo -e "\n${YELLOW}Secret Manager secrets:${NC}"
        gcloud secrets list 2>/dev/null || echo "None or no access"
        
        echo -e "\n${YELLOW}Pub/Sub topics:${NC}"
        gcloud pubsub topics list 2>/dev/null || echo "None or no access"
        
        # Switch back
        gcloud config set project "$ORIGINAL_PROJECT" --quiet
        ;;
        
    5)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo -e "\n✅ Cleanup complete!"