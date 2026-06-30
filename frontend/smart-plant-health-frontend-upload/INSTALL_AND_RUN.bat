@echo off
echo Smart Plant Health - Latest Fixed Source
echo.
npm config set registry https://registry.npmjs.org/
npm install --legacy-peer-deps --no-audit --no-fund
npm run dev
pause
