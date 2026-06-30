# Ecosense AI - Mobile Gallery and Responsive Fixes

## Scope
This update improves the existing Ecosense AI / Smart Plant Health website without creating a new project.

## Completed fixes
- Added a clear mobile image picker with two separate actions:
  - Take Photo: opens the mobile camera using capture=environment.
  - Choose from Gallery: opens the image/file picker without capture and accepts jpg/jpeg/png/webp.
- Added image preview before analysis.
- Added change image and remove image actions.
- Added mobile-friendly helper text and translated labels.
- Improved Diagnosis page responsiveness on phones.
- Improved global mobile layout: cards, grids, forms, buttons, farm tabs, reports, workers, and settings.
- Added stronger mobile overflow protection to prevent elements from entering each other.
- Kept backend diagnosis logic unchanged:
  - predict_sensors sends JSON.
  - predict_image sends form-data with file.
  - predict_with_image sends file + sensor readings.
- Build completed successfully.

## Important note
On mobile browsers, exact camera/gallery behavior depends on Android/iOS browser permissions. The camera button uses capture to request camera directly. The gallery button uses a normal image picker to let the user select an existing file/photo.
