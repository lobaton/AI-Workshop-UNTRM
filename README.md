# AI-Workshop-UNTRM

Material for the AI workshop at the Universidad Nacional Toribio Rodríguez de Mendoza (UNTRM), Chachapoyas, Peru.

| Session | Folder |
|---|---|
| 1. From Data to Agents: A Hands-On Introduction to AI | [`session1_from_data_to_agents`](session1_from_data_to_agents/) |
| 2. Inside the Black Box: Foundations of Deep Learning | [`session2_inside_the_black_box`](session2_inside_the_black_box/) |
| 3. Coding with a Copilot: Adapting Pre-Trained Models for Your Research | [`session3_coding_with_a_copilot`](session3_coding_with_a_copilot/) |
| 4. Seeing with AI: Image Processing from Pixels to Deployment | [`session4_seeing_with_ai`](session4_seeing_with_ai/) |

**Running on the HPC cluster:** [`HPC-Setup`](HPC-Setup/) explains how to connect to the cluster and run
GPU-accelerated Jupyter notebooks from VS Code with GitHub Copilot. It covers the environment setup, the
connection, troubleshooting, and a notebook to check that the GPU works.

**For maintainers:** run `git config core.hooksPath .githooks` once per clone, so each commit refreshes the
saved outputs that the molab previews display (see [`.githooks/README.md`](.githooks/README.md)).
