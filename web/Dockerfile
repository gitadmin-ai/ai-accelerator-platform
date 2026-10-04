# Build from the repo root or from web/ directly: docker build -f web/Dockerfile web/
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Relative: the browser talks to nginx, which proxies /api/ to BACKEND_URL.
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
# Where /api/ is proxied to; override at runtime (e.g. http://localhost:8000 in a shared pod).
ENV BACKEND_URL=http://backend:8000
EXPOSE 80
