import bisect, collections, hashlib, json
from pathlib import Path
SOURCE = Path('/private/tmp/bracketable600.json')
OUT = Path('/private/tmp/overlay2d_visibility60_selected114')

def max_gap(mask):
    longest = 0
    start = None
    for index, valid in enumerate(list(mask) + [True]):
        if not valid and start is None:
            start = index
        elif valid and start is not None:
            longest = max(longest, index - start)
            start = None
    return longest

def missing_runs(mask):
    result = []
    start = None
    for index, valid in enumerate(list(mask) + [True]):
        if not valid and start is None:
            start = index
        elif valid and start is not None:
            result.append({'start_display_index': start, 'end_display_index': index - 1, 'length': index - start})
            start = None
    return result

items = json.loads(SOURCE.read_text())
eligible = [x for x in items if x['per_frame_both_valid'][0] and x['per_frame_both_valid'][-1] and max_gap(x['per_frame_both_valid']) <= 60]
groups = collections.defaultdict(list)
for item in eligible:
    groups[(item['dataset'], item['sequence_id'])].append(item)
selected = []
for key, group in groups.items():
    group.sort(key=lambda x: (x['actual_end_position_exclusive'], x['actual_start_position'], x['dataset_rank']))
    ends = [x['actual_end_position_exclusive'] for x in group]
    dp = [(0, 0.0, [])]
    for index, item in enumerate(group):
        previous = bisect.bisect_right(ends, item['actual_start_position'], 0, index) - 1
        base = dp[previous + 1]
        take = (base[0] + 1, base[1] + float(item['score']), base[2] + [index])
        skip = dp[-1]
        take_ranks = [group[i]['dataset_rank'] for i in take[2]]
        skip_ranks = [group[i]['dataset_rank'] for i in skip[2]]
        choose = take[0] > skip[0] or (take[0] == skip[0] and (take[1] > skip[1] + 1e-12 or (abs(take[1] - skip[1]) <= 1e-12 and take_ranks < skip_ranks)))
        dp.append(take if choose else skip)
    selected.extend(group[i] for i in dp[-1][2])
selected.sort(key=lambda x: (x['dataset'], x['dataset_rank']))
assert len(selected) == 114
OUT.mkdir(exist_ok=False)
manifest = OUT / 'selected_manifest.jsonl'
with manifest.open('x') as handle:
    for item in selected:
        row = dict(item)
        row['selection_status'] = 'visibility60_max_nonoverlap_selected'
        row['maximum_internal_both_hand_missing_run'] = max_gap(row['per_frame_both_valid'])
        row['internal_both_hand_missing_runs'] = missing_runs(row['per_frame_both_valid'])
        handle.write(json.dumps(row, separators=(',', ':')) + '\n')
raw = manifest.read_bytes()
sha = hashlib.sha256(raw).hexdigest()
counts = {}
for dataset in ('arctic','h2o','hoi4d','hot3d','oakink_v2','taco'):
    rows = [x for x in selected if x['dataset'] == dataset]
    counts[dataset] = {'selected': len(rows), 'with_internal_missing': sum(not x['all_frames_both_hands_valid'] for x in rows), 'display_frames': sum(x['actual_length'] for x in rows)}
summary = {
    'schema': 'overlay-2d-visibility60-max-nonoverlap-v1',
    'source_candidates': str(SOURCE),
    'criterion': 'both hands valid at first/last display frame; maximum internal combined missing run <= 60 frames',
    'nonoverlap': 'maximum interval count per dataset+sequence; ties maximize original 2D score then lexicographic dataset ranks',
    'selected': len(selected),
    'selected_manifest_sha256': sha,
    'display_frames': sum(x['actual_length'] for x in selected),
    'full_300_frame_clips': sum(x['actual_length'] == 300 for x in selected),
    'clips_with_internal_missing': sum(not x['all_frames_both_hands_valid'] for x in selected),
    'per_dataset': counts,
}
(OUT / 'selection_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
