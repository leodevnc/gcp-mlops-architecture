# Repository instructions

## Branch cleanup after merging

Treat deleting the temporary work branch as part of completing a pull request. Confirm that GitHub reports the pull request as merged and that the remote branch still points to the recorded pull request head before deleting it from the remote.

Delete the local work branch if its tip matches the merged pull request head or is an ancestor of the target branch. Use the recorded pull request head to verify squash and rebase merges.
