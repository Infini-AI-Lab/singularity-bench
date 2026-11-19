def run(task, workspace, model, budget):
    model.complete([{"role": "user", "content": task}])
