/**
 * PM2 ecosystem config for Poker44 UID 208 (justice-coldkey / justice-hotkey-poker44).
 * Pinned to v123 — R1-era threshold_logit pipeline (Jul 6 benchmark).
 */
module.exports = {
  apps: [
    {
      name: "poker44-miner",
      cwd: "/root/Poker44-top-miner",
      script: "./miner_env/bin/python",
      args: [
        "./neurons/miner.py",
        "--netuid",
        "126",
        "--wallet.name",
        "justice-coldkey",
        "--wallet.hotkey",
        "justice-hotkey-poker44",
        "--subtensor.network",
        "finney",
        "--axon.port",
        "8091",
        "--logging.debug",
        "--blacklist.allowed_validator_hotkeys",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u",
        "5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k",
        "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD",
        "5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126",
        "5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA",
        "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
        "5Hftk9jrMGSJtKBPWkkAkU53FUSr2BqHGPCThg7mbob3hEq1",
        "5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126",
        "5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5",
        "5FLoWCDovMPeH3Gv4syQSZ8TuKcMv6N27g8diDU8zJSeRv8m",
        "5DqrUa2z6E9taJdY8FGiPCrtCswsEjHjPbVo5xcTw2GqvKZm",
      ],
      interpreter: "none",
      env: {
        PYTHONPATH: "/root/Poker44-top-miner",
        POKER44_MODEL_PATH: "/root/Poker44-top-miner/models/poker44_v123_deploy.joblib",
        POKER44_MODEL_NAME: "poker44-v123-hybrid",
        POKER44_MODEL_VERSION: "1.23.0",
        POKER44_MODEL_SHA256: "1954f2610dd4614d034f69c52fa55d834d24c2c8a5fe837cdb76b4af12603598",
        POKER44_MODEL_ARTIFACT_SHA256: "1954f2610dd4614d034f69c52fa55d834d24c2c8a5fe837cdb76b4af12603598",
        POKER44_MODEL_REPO_URL:
          "https://github.com/Yaroslav98214/poker44-handngram-miner.git",
        POKER44_MODEL_REPO_COMMIT: "12c031d6a669829e1fa661e8d5371613db004bd0",
        POKER44_MODEL_OPEN_SOURCE: "true",
        POKER44_MODEL_FRAMEWORK: "hybrid-lgb-xgb-et-hgram-v123-r1",
        POKER44_MODEL_TRAINING_DATA_SOURCES: "released_training_benchmark_v113",
        POKER44_MODEL_TRAINING_DATA_STATEMENT:
          "v123 with Jul 12 live arena calibration retune for homogeneous batch bot recall.",
        POKER44_MODEL_PRIVATE_DATA_ATTESTATION:
          "No private data used. Training uses only the public benchmark API corpus.",
        POKER44_MODEL_DATA_ATTESTATION:
          "No private data used. Training uses only the public benchmark API corpus.",
        POKER44_LOG_SCORE_ARRAYS: "1",
        POKER44_LOG_SCORE_COMPONENTS: "1",
      },
    },
  ],
};
