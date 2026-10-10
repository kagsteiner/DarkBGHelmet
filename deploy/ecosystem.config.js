// PM2 configuration for the DarkBGHelmet HTTP engine.
//   pm2 start deploy/ecosystem.config.js && pm2 save
// The server only listens on 127.0.0.1; your apps call it locally.
// BG_PORT, BG_THREADS, BG_HASH_MB and BG_PLIES override the defaults below.
const env = process.env;

module.exports = {
  apps: [
    {
      name: "bgengine",
      cwd: __dirname + "/..",              // repository root (nets/ and data/ live here)
      script: "./build/bgserver",
      interpreter: "none",                  // native binary, not a Node script
      args: [
        "--host", "127.0.0.1",
        "--port", env.BG_PORT || "8085",
        "--net", "nets/latest.bgnn",
        "--data", "data",
        "--threads", env.BG_THREADS || "2",  // CPU cores one request may use
        "--hash", env.BG_HASH_MB || "64",    // MB for the search cache
        "--plies", env.BG_PLIES || "3",      // default depth (requests may ask for more)
        "--cube-extra", "0",                 // extra plies for cube decisions
      ].join(" "),
      autorestart: true,
      max_memory_restart: "800M",
    },
  ],
};
