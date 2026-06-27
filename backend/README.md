# EcoSense Backend

Node.js + Express + MongoDB backend for EcoSense project.

## Quick start

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create .env file from .env.example and adjust values.

3. Start local MongoDB (ensure `mongod` is running).

4. Run in development:
   ```bash
   npm run dev
   ```

## API endpoints
- POST /api/users/register
- POST /api/users/login
- POST /api/sensors/upload
- GET  /api/sensors/latest
- POST /api/sensors/predict
