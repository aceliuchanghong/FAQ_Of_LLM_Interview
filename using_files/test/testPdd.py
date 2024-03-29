def dfs(nums, target, index, path, result):
    if target == 0:
        result.append(path)
        return
    if target < 0 or index == len(nums):
        return
    # 包含当前元素
    dfs(nums, target - nums[index], index + 1, path + [nums[index]], result)
    # 不包含当前元素
    dfs(nums, target, index + 1, path, result)


def find_subarrays(nums, target):
    result = []
    dfs(nums, target, 0, [], result)
    return result


# 示例用法
nums = [1, 1, 2, 3, 5]
target = 7
print(find_subarrays(nums, target))
