FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 8766
CMD ["node", "a2f_proxy.mjs"]