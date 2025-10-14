## Running Evaluation with Docker

To evaluate hallucination results in a clean, isolated environment:

Build the Docker image:
```bash
cd docker
docker compose build
```

Start a container shell:
```bash
docker compose run codehalu-eval /bin/bash
```

Run the evaluation manually inside the container:

```bash
python3 eval.py --halu_type Logic_Deviation --generation_file foo.jsonl
```

Full list of hallucination types: 
- `Data_Compliance_Hallucination`
- `Structural_Access_Hallucination`
- `Identification_Hallucination`
- `External_Source_Hallucination`
- `Physical_Constraint_Hallucination`
- `Calculate_Boundary_Hallucination`
- `Logic_Deviation`
- `Logic_Breakdown`

View results:
Evaluated outputs and error logs will be saved automatically in
`evaluated_results`
