#!/usr/bin/env python3
"""
Python测试脚本 - 验证lattice_env模块
"""

import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "lib")
sys.path.insert(0, build_dir)

def test_import():
    """测试模块导入"""
    print("=" * 50)
    print("测试 lattice_env 模块导入")
    print("=" * 50)
    
    try:
        import lattice_env
        print(f"? 成功导入 lattice_env 版本: {lattice_env.__version__}")
        return lattice_env
    except ImportError as e:
        print(f"? 导入失败: {e}")
        print("\n可能的原因:")
        print("1. 模块未编译 - 请先运行 build.sh")
        print("2. Python版本不匹配")
        print("3. 模块路径错误")
        return None

def test_create_lattice(module):
    """测试创建Lattice对象"""
    print("\n" + "=" * 50)
    print("测试创建Lattice对象")
    print("=" * 50)
    
    try:
        # 创建格对象
        lattice = module.create_lattice(40, 40)
        print("? 成功创建 Lattice 对象")
        
        # 注意：这里需要调用一些方法来初始化格
        # 例如：lattice.setSVPChallenge(40, 0)
        # 但需要先在C++端暴露这些方法
        
        return lattice
    except Exception as e:
        print(f"? 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_create_env(module, lattice):
    """测试创建环境"""
    print("\n" + "=" * 50)
    print("测试创建环境")
    print("=" * 50)
    
    try:
        env = module.LatticeEnv(lattice)
        print("? 成功创建 LatticeEnv 对象")
        
        # 测试配置
        config = module.Config()
        config.max_dimension = 40
        config.action_range = 5.0
        config.max_steps = 2000
        env.set_config(config)
        print("? 成功设置配置")
        
        return env
    except Exception as e:
        print(f"? 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reset_and_step(env):
    """测试reset和step"""
    print("\n" + "=" * 50)
    print("测试环境交互")
    print("=" * 50)
    
    try:
        # 重置环境
        state = env.reset(R=100.0)
        print(f"? 重置成功，状态维度: {len(state)}")
        print(f"   环境维度: {env.dimension}")
        
        # 执行几步
        for i in range(5):
            # 随机动作（-5到5）
            action = 0  # 先测试动作0
            
            state, reward, done, info = env.step(action)
            print(f"  步 {i+1}: 奖励={reward:.4f}, 完成={done}, 信息={info}")
            
            if done:
                print(f"  提前结束: {info}")
                break
        
        print(f"\n? 环境交互测试完成")
        print(f"   当前k: {env.current_k}")
        print(f"   当前rho: {env.current_rho}")
        print(f"   已解决: {env.solved}")
        
    except Exception as e:
        print(f"? 交互测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("开始 lattice_env 模块测试")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print(f"构建目录: {build_dir}")
    
    # 1. 测试导入
    module = test_import()
    if not module:
        return
    
    # 2. 测试创建Lattice
    lattice = test_create_lattice(module)
    if not lattice:
        return
    
    # 3. 测试创建环境
    env = test_create_env(module, lattice)
    if not env:
        return
    
    # 4. 测试交互
    test_reset_and_step(env)
    
    print("\n" + "=" * 50)
    print("? 所有测试完成！")
    print("=" * 50)
    print("\n下一步:")
    print("1. 修改 lattice.h 添加友元类")
    print("2. 在C++中暴露更多Lattice方法")
    print("3. 实现完整的ENUM逐步执行")

if __name__ == "__main__":
    main()