import json


def load_and_execute_functions(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 创建一个字典来保存执行后的函数
    global_namespace = {}

    for func_info in data["functions"]:
        code_str = func_info["code"]
        # print(f"Executing function:\n{code_str}\n{'-'*40}")
        exec(code_str, global_namespace)

    return global_namespace


if __name__ == "__main__":
    """
    cd using_files/Reward_Guided_Code_Optimizer/
    python test_corpus.py
    """
    functions = load_and_execute_functions("function_corpus.json")

    print(functions["factorial"](5))
    print(functions["fibonacci"](6))
    print(functions["sum_list"]([1, 2, 3]))
    print(functions["reverse_string"]("hello"))
    print(functions["is_palindrome"]("Madam"))
    print(functions["max_in_list"]([3, 7, 2, 9, 5]))
    print(functions["count_vowels"]("Beautiful Day"))
    print(functions["power"](2, 5))
    print(functions["unique_elements"]([1, 2, 2, 3, 1, 4]))
    print(functions["sort_list"]([5, 3, 8, 1, 2]))
