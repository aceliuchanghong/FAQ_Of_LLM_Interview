# pip install easy-media-utils
from tree_utils.struct_tree_out import print_tree

path = r'../../../FAQ_Of_LLM_Interview'
exclude_dirs_set = {'using_files', 'LangChain', 'pytorch', 'thoughts_on_llm'}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
