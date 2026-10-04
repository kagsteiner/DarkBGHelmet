// PM2 configuration for the DarkBGHelmet HTTP engine.
//   pm2 start deploy/ecosystem.config.js && pm2 save
// The server only listens on 127.0.0.1; your Node app calls it locally.
module.exports = {
  apps: [
    {
      name: "bgengine",
      cwd: __dirname + "/..",              // repository root (nets/ and data/ live here)
      script: "./build/bgserver",
      interpreter: "none",                  // native binary, not a Node script
      args: [
        "--host", "127.0.0.1",
        "--port", "8085",
        "--net", "nets/latest.bgnn",
        "--data", "data",
        "--threads", "2",                   // number of CPU cores to use per request
        "--hash", "64",                     // MB for the search cache
        "--plies", "3",                     // default depth (requests may ask for more)
        "--cube-extra", "0",                // extra plies for cube decisions
      ].join(" "),
      autorestart: true,
      max_memory_restart: "800M",
    },
  ],
};
