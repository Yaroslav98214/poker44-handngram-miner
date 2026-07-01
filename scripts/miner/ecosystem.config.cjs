/**
 * PM2 ecosystem config for Poker44 UID 198 (justice-coldkey / justice-hotkey-poker44).
 *
 * Usage:
 *   pm2 start scripts/miner/ecosystem.config.cjs
 *   pm2 save
 *
 * After a git push, update POKER44_MODEL_REPO_COMMIT to match `git rev-parse HEAD`
 * before R6 opens so the manifest matches the published repo.
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
        POKER44_MODEL_PATH: "/root/Poker44-top-miner/models/poker44_v122_deploy.joblib",
        POKER44_MODEL_NAME: "poker44-v122-hybrid",
        POKER44_MODEL_VERSION: "1.22.0",
        POKER44_MODEL_SHA256:
          "e3a556640502bd08c844056e2a0e0f55e231ddc8c7f0aa34e545b9a40c8d15e7",
        POKER44_MODEL_ARTIFACT_SHA256:
          "e3a556640502bd08c844056e2a0e0f55e231ddc8c7f0aa34e545b9a40c8d15e7",
        POKER44_MODEL_REPO_URL:
          "https://github.com/Yaroslav98214/poker44-handngram-miner.git",
        POKER44_MODEL_REPO_COMMIT:
          "09b61b8658fe2820d39674429c433129d0027bcd",
        POKER44_MODEL_OPEN_SOURCE: "true",
        POKER44_MODEL_FRAMEWORK: "hybrid-lgb-xgb-et-hgram-quantile-apfirst",
        POKER44_MODEL_TRAINING_DATA_SOURCES: "released_training_benchmark_v112",
        POKER44_MODEL_TRAINING_DATA_STATEMENT:
          "Trained on public Poker44 benchmark through 2026-07-01 with AP-first calibration and live-range score_remap guard for mixed R5/R6 batches.",
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
