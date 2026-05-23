# syntax=docker/dockerfile:1.7
# ───── Self-Learning Coding AI · frontend image ─────
# Builds the Vite static bundle and serves it via the dev server.
# (For prod-grade serving, build → copy `dist/` into nginx image.)

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json ./
# Use npm install (no lockfile checked in) — the dev image regenerates one.
RUN npm install --no-audit --no-fund

FROM node:20-alpine AS dev
WORKDIR /app
ENV NODE_ENV=development
COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 5173

# Vite dev server. For prod, replace with: `RUN npm run build` and serve
# /app/dist with nginx.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
