# Greedy, Auto-Dateiname data/compare_<item>.vtk
uv run python scripts/compare_viz.py --tokens data/hexarow_tokens_family_cart.pt  --ckpt data/grpo_cart_step300.pt --idx 687

# Stochastisch, k Rollouts -> <stem>_r<i>.vtk je Dateiname
uv run python scripts/compare_viz.py --tokens data/hexarow_tokens_family_cart.pt --ckpt data/grpo_cart_step300.pt --idx 683 --temperature 0.7 --k 4 --out data/gen_test.vtk
