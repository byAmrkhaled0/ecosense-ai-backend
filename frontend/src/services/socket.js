import { io } from "socket.io-client";
import { getAuthToken } from "./api";

const SOCKET_URL =
  import.meta.env.VITE_SOCKET_URL || "https://ecosense-backend.vercel.app";

// Socket is optional. It is disabled by default to avoid endless failed websocket retries
// and 429 side effects when the deployed backend/socket service is not available.
const SOCKET_ENABLED = String(import.meta.env.VITE_ENABLE_SOCKET || "false").toLowerCase() === "true";

let socket = null;
let joinedUserId = null;
let connecting = false;

export const connectSocket = (userId) => {
  if (!SOCKET_ENABLED) return null;

  if (socket?.connected) {
    if (userId && userId !== joinedUserId) {
      socket.emit("join", userId);
      joinedUserId = userId;
    }
    return socket;
  }

  if (connecting && socket) return socket;
  connecting = true;

  const token = getAuthToken();

  socket = io(SOCKET_URL, {
    transports: ["polling"],
    reconnection: true,
    reconnectionAttempts: 1,
    reconnectionDelay: 8000,
    timeout: 8000,
    auth: token ? { token } : undefined,
    extraHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
  });

  socket.on("connect", () => {
    connecting = false;
    if (userId) {
      socket.emit("join", userId);
      joinedUserId = userId;
    }
  });

  socket.on("disconnect", () => {
    connecting = false;
    joinedUserId = null;
  });

  socket.on("connect_error", () => {
    connecting = false;
  });

  return socket;
};

export const getSocket = () => socket;

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
    joinedUserId = null;
    connecting = false;
  }
};

export const onNotification = (callback) => {
  if (!socket) return () => {};

  socket.on("newNotification", callback);

  return () => {
    socket.off("newNotification", callback);
  };
};
