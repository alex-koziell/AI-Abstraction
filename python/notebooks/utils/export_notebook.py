import json, fire, re
from pathlib import Path
import io

def is_export_cell(cell):
    """ Returns whether a cell is a block of code that
        has been labelled for export. """
    
    if cell['cell_type'] != 'code':
        return False
    
    src = cell['source']
    if len(src) == 0 or len(src[0]) < 7:
        return False
    
    return bool(re.match(r' *\#-+export-+\# *$', src[0], re.IGNORECASE))


def get_notebooks(file_dir):
    """ Returns all notebooks in a directory. """

    from glob import glob
    
    notebooks = []
    if isinstance(file_dir, str):
        notebooks = glob(file_dir + '*.ipynb')
    if len(notebooks) == 0:
        print('WARNING: No files found in directory.')
        return notebooks
    return sorted(notebooks)


def export_notebook(file_path=None, file_dir=None):
    """ Runs cells_to_script on a notebook or directory of notebooks. """ 

    if (not file_path) and (not file_dir):
        print('No files or folders provided!')

    if not file_dir:
        cells_to_script(file_path)
    else:
        [cells_to_script(notebook) for notebook in get_notebooks(file_dir)]
        print('All notebooks converted!')


def cells_to_script(file_path):
    """ Finds cells starting with `#export`
        and puts them in a new module. """
    
    file_path = Path(file_path)
    
    notebook_dict = json.load(open(file_path,
                                     'r', encoding='utf-8'))
    
    code_cells = [cell for cell in notebook_dict['cells']
                  if is_export_cell(cell)]
    module_code = f'# module automatically generated from {file_path.name}\n\n'

    for cell in code_cells:
        module_code += ''.join(cell["source"][1:]) + '\n\n'
    # remove trailing whitespace
    re.sub(r' +$', '', module_code, flags=re.MULTILINE)

    if not (file_path.parent/'exports').exists():
        (file_path.parent/'exports').mkdir()
    output_path = f'{file_path.parent}/exports/e_{file_path.stem}.py'
    
    with io.open(output_path, 'w', encoding='utf-8') as ofile:
        ofile.write(module_code[:-2])
    print(f'Notebook {file_path.name} has been converted to module {output_path}!')


if __name__ == '__main__':
    fire.Fire(export_notebook)
