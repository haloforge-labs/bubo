# Start your first Bubo server

Complete the [installation steps](install.md) first.
Use the `default` profile for normal use.
It selects medium effort and adaptive thinking for Qwen3.8 Flash Next.


## Select the files

Run `bubo prepare` first (see the [installation steps](install.md)). The prepare step writes `bubo-package.json` beside the model. The server finds the dense twin, the draft head, the MTP predictor and the vision projector through that file. You do not set file paths.

1. Open PowerShell in the package folder.
2. Set the path to the first model shard.

```powershell
$model = "$env:LOCALAPPDATA\Bubo\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
```

If you used `--output ROOT`, replace `$env:LOCALAPPDATA\Bubo` with `ROOT`.
A flag or an environment value overrides the manifest. A broken manifest or a missing named file stops the server and names the file.

## Start the optimal text setup

1. Start the server.

```powershell
.\bubo.exe serve llm --model $model --served-model-name bubo
```

The `bubo` model identifier is explicit so the API example can use a fixed name.
The default profile supplies survival control, automatic draft vocabulary, and prompt lookup with MTP.
It enables bounded lenient acceptance inside thinking spans.
It uses one session, 262144 context tokens, and a 6 GiB RAM cache.
Disk prompt cache stays off.

2. Wait for `event=load_completed` in the log.

## Send a first request

1. Open a second PowerShell window.
2. Set the request body.

```powershell
$body = @{
    model = "bubo"
    messages = @(@{ role = "user"; content = "Explain what a local language model does in two sentences." })
    max_tokens = 512
} | ConvertTo-Json -Depth 5
```

3. Send the request.

```powershell
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/v1/chat/completions" -ContentType "application/json" -Body $body
```

4. Display the answer.

```powershell
$response.choices[0].message.content
```

The server speaks the OpenAI API. Point any OpenAI-compatible client, such as a coding agent, at the base URL `http://127.0.0.1:8080/v1` and use the model name `bubo`.
Keep the local address for normal laptop use.

## Images

The vision projector from the manifest loads by default. Images need more memory.
To save memory when you do not need images, start the server with `--no-vision`.

```powershell
.\bubo.exe serve llm --model $model --no-vision --served-model-name bubo
```

## Inspect the settings

1. Print the effective settings without a model load.

```powershell
.\bubo.exe serve llm --model $model --print-effective-settings
```

This command prints JSON and exits before it starts a server.

## Use exact output

1. Stop the current server with `Ctrl+C`.
2. Start the exact profile.

```powershell
.\bubo.exe serve llm --model $model --profile exact --seed 1 --served-model-name bubo
```

The exact profile leaves out the MTP predictor, the dense twin and the draft head, even when the manifest names them.
It refuses flags that turn on approximation or speculation (exit code 2).
It disables speculation, behaviour policies, and prompt caches.
Each request processes its full prompt.
Keep the prompt and request settings fixed for reproducibility.
See [profiles](profiles.md) for the exact scope.

## Fast restarts with the disk cache

The disk prompt cache is off by default. When you turn it on, Bubo keeps the computed prompt state of long conversations on the disk. After a server restart, or after the RAM cache drops a conversation, the conversation continues without processing the whole prompt again.

Measured on a 70 W Strix Halo laptop with a 30K-token session:

| Measured | Value |
| --- | --- |
| Cold prompt processing, 30K tokens | About 25 s |
| Next turn, same server, from memory | 0.9 s |
| Continue after a server restart, with the disk cache | About 2 s (1.8 to 2.4 s in two runs; about 30,400 prompt tokens restored) |
| First start with the cache (one-time model check) | 74 s |
| Later starts with the cache | 29 to 32 s |
| Size of one saved snapshot | About 954 MB for 30K tokens (about 31 KB for each token) |

In an earlier test, a 109K-token session continued after a restart in 3.5 s instead of 88.8 s. A long saved session writes about 3 GB to the SSD. A 90-request agent replay wrote about 5.1 GB. Decode speed cost is about 2.5 %. The cache writes to your SSD. Use these numbers to decide if the cache is right for your disk.

To turn it on, start Bubo with the folder for the cache:

```powershell
start.cmd -CacheDisk D:\BuboCache
```

The first start with the cache reads the eight model files once and checks them. Bubo saves a stamp. Later starts use the stamp and do not read the files again. Without a prepared model package the server refuses the cache.

To use the server command directly, add `--cache-disk "D:\BuboCache"` to it. Run `start.cmd -DryRun` to print the command that `start.cmd` runs.

- Bubo saves a snapshot about 30 s after the conversation goes idle.
- The disk budget is 8 GiB by default. Change it with `--cache-disk-bytes`. A long session can need more.
- `start.cmd -CacheDisk <folder>` is all you need. It also sets a 5 GiB staging area. Change the staging size with `-CacheDiskStagingBytes`.
- If you run `bubo serve` directly, add `--cache-disk-staging-bytes 5368709120` for long sessions. Bubo stages each snapshot in RAM before it writes it. The automatic staging size follows the free RAM at start, and it can be too small for one snapshot. With about 7 GiB free, the area was 941 MB, smaller than one 30K-token snapshot (954 MB). Bubo then dropped the snapshots and the restart processed the whole prompt again (25 s).
- Turn the cache off again with `--no-cache-disk`. The `exact` profile turns all caches off and ignores the disk cache.


See [options](options.md) for other controls.
