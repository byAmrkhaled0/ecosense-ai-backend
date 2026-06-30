# Ecosense Sector Final Small Boxes Fixes

## Done
- Reordered Sector Details layout into compact columns:
  - Equipment → Sector Sensors → Workers table
  - Latest Diagnosis → Care Tasks → Notes
- Workers inside Sector Details are now a small compact table for assigned sector workers only.
- The full Owner permissions table is kept in the Admin Panel page, not inside Sector Details.
- Added compact permissions table in Admin Panel with View / Add / Edit / Delete permissions.
- Further reduced padding, height, gaps, and inner spacing of Sector Details cards.
- Strengthened Camera Card pointer-events/z-index so Refresh and Analyze buttons stay clickable.
- Kept Ask Assistant button at the bottom of the Sector page.

## Build note
npm install timed out in this environment, so build could not be completed here. The package is flat and ready to run locally with npm install and npm run build.
