# Ecosense AI Final Polish Fixes

This delivery continues on the same Ecosense AI / Smart Plant Health project without creating a new template or project.

## Fixed
- Added a final translation fallback to prevent raw translation keys from appearing in the UI.
- Added missing English and Arabic labels for report names, disease guide items, sector names, sensor backend states, and login/security copy.
- Cleaned report names so cards show clear names such as Diagnosis Report, Sensor Report, Farm Status Report, and Disease Alert Report.
- Updated Disease Guide to display fully localized disease names, symptoms, causes, initial solutions, and prevention steps in both English and Arabic.
- Improved Farm Management tabs on mobile as clean scrollable segmented chips without visible scrollbar.
- Improved Assistant on mobile as a compact bottom-sheet style panel instead of a large popup that covers content.
- Disabled automatic sensor polling when the sensor backend endpoint is not available, showing Simulation Mode / Backend Required messaging instead.
- Improved Login permission copy spacing and translation coverage.
- Added stronger mobile CSS guards for reports, diagnosis, disease guide, assistant, and farm management.

## Build
- npm run build completed successfully.
- Only the Vite large chunk warning remains. It is not a build error.

## Modified Files
- src/App.jsx
- src/index.css
- dist/
- ECOSENSE_FINAL_POLISH_KEYS_REPORTS_MOBILE_FIXES.md
