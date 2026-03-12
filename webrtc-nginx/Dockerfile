# Build context: repo root
FROM nginx:alpine

# Bake static frontend files into the image for production.
# In dev mode this layer is bypassed via a bind-mount (webrtc-docker-compose.dev.yml).
COPY webrtc-client/ /usr/share/nginx/html/
COPY webrtc-nginx/conf.d/  /etc/nginx/conf.d/
