# What's new in 0.1.1

- **The Bubo app.** Pick a model, press Start, and watch memory and requests. Update Bubo with one click.
- **Faster.** Writing is up to 11 % faster than in 0.1.0 on mixed work, and 10 % faster on code edits. Reading a 32K prompt is 5 % faster.
- **A shorter wait for the first token.** At 100K tokens of context the first token arrives about 113 ms sooner.
- **More models.** Qwen3.8 27B, Qwen3.6 27B, Cyber-Tiel Coder, Gemma 4 26B and 31B. They take images.
- **Less memory.** Choose a smaller model, a shorter context, or the lower-memory build of Qwen3.8 Flash Next. Bubo keeps 4 GiB of memory free and turns away a request that does not fit, so the PC keeps running.
- **Repair.** If a model file is damaged, Bubo finds it and replaces it after you approve.
- **Gemma 4 builds.** Gemma 4 26B-A4B and 31B each come in two builds: Unsloth QAT and Huihui abliterated QAT. With the draft model on, medium-length replies on the Huihui builds are about 1.1x faster on 26B-A4B and about 1.9x faster on 31B.
- **DiffusionGemma takes images.** DiffusionGemma 26B E38 is in the model list. Bubo downloads its vision file together with the model.
- **96 GB machines.** Use the lower-memory build of Qwen3.8 Flash Next. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.
- **Memory check.** Bubo checks memory before Start. A tight setup starts only after you tick the override for it; a setup that does not fit never starts.
- **Live progress.** `bubo status --watch` and the Requests column show what the server is doing now.
- **Smaller updates.** An update keeps the files that did not change.
