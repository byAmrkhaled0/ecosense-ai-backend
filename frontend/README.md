# EcoSense AI Frontend

Production-ready frontend for EcoSense AI Smart Plant Health.

## Local run

```bash
npm install
npm run dev
```

Open: http://127.0.0.1:3000/login

## Production build

```bash
npm run build
```

## Environment

Local `.env.development` uses Vite proxy:

```env
VITE_API_BASE_URL=/api
VITE_USE_DEV_PROXY=true
VITE_API_WITH_CREDENTIALS=false
VITE_API_TIMEOUT=15000
VITE_SOCKET_URL=https://ecosense-backend.vercel.app
VITE_GOOGLE_CLIENT_ID=
```

Production `.env.production` uses the real backend:

```env
VITE_API_BASE_URL=https://ecosense-backend.vercel.app/api
VITE_USE_DEV_PROXY=false
VITE_API_WITH_CREDENTIALS=false
VITE_API_TIMEOUT=15000
VITE_SOCKET_URL=https://ecosense-backend.vercel.app
VITE_GOOGLE_CLIENT_ID=
```
