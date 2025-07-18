#!/bin/bash

# Delete old test projects (keeping the main Athena projects)
echo "🧹 Deleting old test projects..."

# List of old projects to delete (excluding Athena projects)
OLD_PROJECTS=(
    "abhishek-959"
    "abhisonu-6c0bd"
    "abhisonu12-a1a07"
    "abi-bagel"
    "abiproj-8e4a3"
    "nolymit-4b031"
    "celestial-tract-379304"
    "gen-lang-client-0587760709"
    "chaintales"
)

CURRENT_PROJECT=$(gcloud config get-value project)

for project in "${OLD_PROJECTS[@]}"; do
    if [ "$project" != "$CURRENT_PROJECT" ]; then
        echo "Deleting project: $project"
        gcloud projects delete "$project" --quiet || echo "Failed to delete $project"
    fi
done

echo "✅ Cleanup complete!"
echo "Remaining projects:"
gcloud projects list --format="table(projectId,name,createTime)"