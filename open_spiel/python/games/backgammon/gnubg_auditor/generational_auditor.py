import sys
import os
import re
import subprocess
import statistics
from google.cloud import storage

BUCKET_NAME = "expert-eyes-training-742"

def clean_legacy_log(content):
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        match = re.search(r"^\s*(\d+)\)\s+(\d)(\d): (.*?)\s{2,}(\d)(\d): (.*)$", line)
        if match:
            idx, d1, d2, m1, d3, d4, m2 = match.groups()
            def expand(m):
                return re.sub(r"(\S+)\((\d)\)", lambda sub: (sub.group(1) + " ") * int(sub.group(2)), m).strip()
            if d1 != d2: m1 = expand(m1)
            if d3 != d4: m2 = expand(m2)
            new_line = f"  {idx:>2}) {d1}{d2}: {m1:<28} {d3}{d4}: {m2}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def parse_gnubg_stats(stdout):
    luck_section = re.search(r"Luck statistics.*?(?=Cube statistics|$)", stdout, re.DOTALL | re.IGNORECASE)
    overall_section = re.search(r"Overall statistics.*?(?=Final score|$)", stdout, re.DOTALL | re.IGNORECASE)
    combined_text = (luck_section.group(0) if luck_section else "") + (overall_section.group(0) if overall_section else "")
    if not combined_text:
        combined_text = stdout

    patterns = {
        'checker_error': r"Error rate mEMG \(MWC\)\s+(-?\d+\.\d+)\s+\(.*?%\)\s+(-?\d+\.\d+)",
        'snowie_error': r"Snowie error rate\s+(-?\d+\.\d+)\s+\(.*?%\)\s+(-?\d+\.\d+)",
        'move_count': r"Total moves\s+(\d+)\s+(\d+)"
    }
    
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, combined_text, re.DOTALL)
        if not match:
            match = re.search(pattern, stdout, re.DOTALL)
        if match:
            parsed[key] = match.groups()
        else:
            parsed[key] = [0.0, 0.0]

    results = []
    for i in range(2):
        results.append({
            'checker_error': float(parsed['checker_error'][i]),
            'snowie_error': float(parsed['snowie_error'][i]),
            'moves': int(parsed['move_count'][i])
        })
    return results

def analyze_game(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    cleaned_content = clean_legacy_log(content)
    with open(file_path, 'w') as f:
        f.write(cleaned_content)

    cmd = ["/usr/games/gnubg", "-t", "-q"]
    stdin_content = (
        f"import auto {file_path}\n"
        "set analysis evaluation plies 2\n"
        "analyze match\n"
        "show statistics match\n"
        "quit\n"
    )
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=stdin_content, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {file_path}")
        return None

    return parse_gnubg_stats(stdout)

def fetch_and_evaluate(gen_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    os.makedirs("audit_temp", exist_ok=True)
    
    types = ["Greedy", "MCTS"]
    prefixes = [f"logs/game_diag_gen{gen_name}_no_mcts_", f"logs/game_diag_gen{gen_name}_mcts_"]
    
    all_game_rows = []
    summary_data = {}
    
    for t_idx, run_type in enumerate(types):
        blobs = list(client.list_blobs(bucket, prefix=prefixes[t_idx]))
        xg_blobs = [b for b in blobs if b.name.endswith("_xg.txt")]
        # Sort sequentially
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        xg_blobs.sort(key=lambda b: natural_sort_key(b.name))
        
        # Take up to 10
        xg_blobs = xg_blobs[:10]
        print(f"Found {len(xg_blobs)} logs for {run_type}.")
        
        checker_errors = []
        snowie_errors = []
        moves_list = []
        
        for g_idx, blob in enumerate(xg_blobs):
            local_path = f"audit_temp/temp_{run_type}_{g_idx}.txt"
            blob.download_to_filename(local_path)
            
            res = analyze_game(os.path.abspath(local_path))
            os.remove(local_path)
            
            if res:
                for p_idx, obs in enumerate(res):
                    ce = obs['checker_error']
                    se = obs['snowie_error']
                    mv = obs['moves']
                    
                    checker_errors.append(ce)
                    snowie_errors.append(se)
                    moves_list.append(mv)
                    
                    # Name / Type / Game / Player / Checker Error Rate / Snowie Error Rate / Moves
                    row = f"{gen_name} / {run_type} / {g_idx} / P{p_idx+1} / {ce:.1f} / {se:.1f} / {mv}"
                    all_game_rows.append(row)
                    print(f"  Processed: {row}")
            else:
                print(f"  Failed to analyze {blob.name}")
                
        if checker_errors:
            summary_data[run_type] = {
                'ce_avg': statistics.mean(checker_errors),
                'ce_med': statistics.median(checker_errors),
                'se_avg': statistics.mean(snowie_errors),
                'se_med': statistics.median(snowie_errors),
                'mv_avg': int(statistics.mean(moves_list)),
                'mv_med': int(statistics.median(moves_list))
            }
        else:
            summary_data[run_type] = None

    # Write Run File
    run_filename = f"audit_run_{gen_name}.txt"
    with open(run_filename, "w") as f:
        f.write("Name / Type / Game / Player / Checker Error Rate / Snowie Error Rate / Moves\n")
        for row in all_game_rows:
            f.write(row + "\n")
        f.write("\nName / Type / Checker Error Rate Average / Checker Error Rate Median / Snowie Error Rate Average / Snowie Error Rate Median / Moves Average / Moves Median\n")
        
        for run_type in types:
            s = summary_data.get(run_type)
            if s:
                summary_row = f"{gen_name} / {run_type} / {s['ce_avg']:.1f} / {s['ce_med']:.1f} / {s['se_avg']:.1f} / {s['se_med']:.1f} / {s['mv_avg']} / {s['mv_med']}"
                f.write(summary_row + "\n")
                
    print(f"\nSaved {run_filename}")
    
    # Upload run file to GCS
    blob = bucket.blob(f"summaries/{run_filename}")
    blob.upload_from_filename(run_filename)
    print(f"Uploaded {run_filename} to GCS.")

    # Master Summary
    master_filename = "master_audit_summary.txt"
    if not os.path.exists(master_filename):
        # Try to download existing master from GCS
        master_blob = bucket.blob(f"summaries/{master_filename}")
        if master_blob.exists():
            master_blob.download_to_filename(master_filename)

    master_row_parts = []
    for run_type in types:
        s = summary_data.get(run_type)
        if s:
            master_row_parts.append(f"{gen_name} / {run_type} / {s['ce_avg']:.0f} / {s['ce_med']:.0f} / {s['se_avg']:.0f} / {s['se_med']:.0f} / {s['mv_avg']} / {s['mv_med']}")
            
    if master_row_parts:
        # Align them with spacing as requested
        # e.g. "2.0 / Greedy / ...         2.0 / MCTS / ..."
        master_line = "         ".join(master_row_parts)
        with open(master_filename, "a") as f:
            f.write(master_line + "\n")
        print(f"\nAppended to {master_filename}")
        
        master_blob = bucket.blob(f"summaries/{master_filename}")
        master_blob.upload_from_filename(master_filename)
        print(f"Uploaded {master_filename} to GCS.")
    else:
        print("No valid summary data to append to master.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generational_auditor.py <GEN_NAME>")
        print("Example: python3 generational_auditor.py 2.0")
        sys.exit(1)
        
    gen_name = sys.argv[1]
    print(f"=== Starting Generational Audit for Gen {gen_name} ===")
    fetch_and_evaluate(gen_name)
