from tree_utils.struct_tree_out import print_tree

path = r'../../'
exclude_dirs_set = {'using_files', 'LangChain'}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
