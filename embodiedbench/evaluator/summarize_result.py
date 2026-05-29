import os
import json
import glob
import argparse

def average_json_values(json_dir, target_file='*.json', output_file='summary_all.json', selected_key=None):
    # We'll accumulate numeric leaves in nested dicts. values_sum/counts will mirror
    # the structure of the numeric fields we encounter.
    values_sum = {}
    counts = {}

    def _add_numeric(keys, num):
        # keys: tuple of path elements; create nested dicts as needed
        d = values_sum
        c = counts
        for k in keys[:-1]:
            d = d.setdefault(k, {})
            c = c.setdefault(k, {})
        last = keys[-1]
        if last not in d:
            d[last] = 0.0
            c[last] = 0
        d[last] += float(num)
        c[last] += 1

    def _process_value(prefix_keys, value):
        # Recursively walk dicts; for lists, unwrap singletons; accept ints/floats
        if isinstance(value, dict):
            for k, v in value.items():
                _process_value(prefix_keys + (str(k),), v)
            return
        if isinstance(value, list):
            if len(value) == 1:
                _process_value(prefix_keys, value[0])
            else:
                # skip lists with multiple elements (ambiguous aggregation)
                return
            return
        if isinstance(value, (int, float)):
            _add_numeric(prefix_keys, value)
            return
        # Non-numeric or string values are ignored
        return

    json_files = glob.glob(os.path.join(json_dir, target_file)) + glob.glob(os.path.join(json_dir, '*', target_file)) + glob.glob(os.path.join(json_dir, '*', '*', target_file))
    print(json_files, len(json_files))
    for json_file in json_files:
        try:
            print(json_file.split('running/')[1])
        except Exception:
            print(json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(data[selected_key] if selected_key is not None else data)
            # if selected_key provided, focus on that sub-object
            if selected_key is not None:
                if selected_key in data:
                    _process_value((str(selected_key),), data[selected_key])
                else:
                    # skip files that don't have the selected key
                    continue
            else:
                for key, value in data.items():
                    # skip strings at top level
                    if isinstance(value, str):
                        continue
                    _process_value((str(key),), value)

    def _compute_averages(sum_node, count_node):
        result = {}
        for k, v in sum_node.items():
            if isinstance(v, dict):
                # recurse
                result[k] = _compute_averages(v, count_node.get(k, {}))
            else:
                cnt = count_node.get(k, 0)
                if cnt > 0:
                    result[k] = v / cnt
                else:
                    result[k] = None
        return result

    averages = _compute_averages(values_sum, counts)
    print('final results: ')
    print(averages)
    with open(os.path.join(json_dir, output_file), 'w') as f:
        json.dump(averages, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files to compute average values.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing JSON files')
    parser.add_argument('--target_file', default='*.json', type=str, help='target file name')
    parser.add_argument('--output_file', default='summary_all.json', type=str, help='output file name')
    args = parser.parse_args()

    average_json_values(args.directory, args.target_file, args.output_file)
