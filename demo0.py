"""
完整的 MNN RAG 系统
集成文档管理、命令解析和智能检索
核心特性：
- 多格式文档导入支持（TXT, Markdown, DOCX, PDF）
- 智能命令解析和验证
- 会话级向量缓存（系统退出时清除）
- 向量索引和缓存持久化
"""
import os
import sys
import time
import json
import numpy as np
from typing import List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import io
import logging

from document_manager import DocumentManager
from command_parser import CommandParser, CommandValidator, CommandType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import MNN.llm as llm
    from sentence_transformers import SentenceTransformer
    import faiss

    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"警告：Faiss不可用 ({e})，将使用基础检索方法")
    FAISS_AVAILABLE = False
    import MNN.llm as llm
    from sentence_transformers import SentenceTransformer


class RAGSystem:
    """增强的 RAG 系统"""

    def __init__(self):
        """初始化系统"""
        # 配置路径
        self.config = {
            'llm_config': r"D:\MNN_RAG_Project\mnn_model\qwen2-1.5b\Qwen2-1.5B-Instruct-MNN\config.json",
            'bge_model': r"D:\MNN_RAG_Project\models\bge-m3",
            'knowledge_base': r"D:\MNN_RAG_Project\src\KB_demo.txt",
            'documents_dir': r"D:\MNN_RAG_Project\documents",
        }

        # 初始化管理器
        self.document_manager = DocumentManager(chunk_size=200, overlap=50)
        self.command_parser = CommandParser()
        self.command_validator = CommandValidator(self.command_parser)

        # 初始化变量
        self.llm_model = None
        self.embedder = None

        # ⭐ 核心数据结构
        self.base_knowledge_fragments = []  # 原始知识库（不变）
        self.knowledge_fragments = []  # 当前知识库（包含文档）
        self.fragment_embeddings = None  # 当前向量
        self.faiss_index = None  # Faiss索引

        # ⭐ 会话状态标记
        self.has_loaded_documents = False  # 是否加载过文档

        # 启动系统
        self.startup()

    def startup(self):
        """启动系统"""
        self.display_welcome()
        self.validate_files()
        self.load_models()
        self.load_knowledge_base()
        self.build_faiss_index()
        self.display_system_info()
        self.start_chat_session()

    def display_welcome(self):
        """显示欢迎信息"""
        print("\n" + "=" * 80)
        print(" " * 25 + "🚀 MNN RAG 系统 ")
        if FAISS_AVAILABLE:
            print(" " * 15 + "LLM: Qwen2-1.5B | Embedding: BGE-M3 | Index: Faiss")
        else:
            print(" " * 15 + "LLM: Qwen2-1.5B | Embedding: BGE-M3 | Index: Basic")
        print("=" * 80 + "\n")

    def validate_files(self):
        """验证文件存在"""
        print("🔍 检查必要文件...")
        all_exist = True

        for key, path in self.config.items():
            if key == 'documents_dir':  # 文档目录可选
                continue
            status = "✅" if os.path.exists(path) else "❌"
            print(f"  {status} {key}: {path}")
            if not os.path.exists(path):
                all_exist = False

        if not all_exist:
            print("\n❌ 一个或多个文件不存在，请检查路径配置")
            sys.exit(1)

        # 创建文档目录（如果不存在）
        if not os.path.exists(self.config['documents_dir']):
            os.makedirs(self.config['documents_dir'], exist_ok=True)
            print(f"  ✅ 文档目录已创建: {self.config['documents_dir']}")

        print()

    def load_models(self):
        """加载模型"""
        print("📦 正在加载模型...")

        # 加载LLM
        print("\n[1/2] 加载Qwen2-1.5B...")
        start_time = time.time()
        try:
            self.llm_model = llm.create(self.config['llm_config'])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.llm_model.load()
            load_time = time.time() - start_time
            print(f"     ✅ 加载完成 ({load_time:.2f}s)")
        except Exception as e:
            print(f"     ❌ LLM加载失败: {e}")
            sys.exit(1)

        # 加载Embedding模型
        print("\n[2/2] 加载BGE-M3嵌入模型...")
        start_time = time.time()
        try:
            self.embedder = SentenceTransformer(self.config['bge_model'])
            load_time = time.time() - start_time
            print(f"     ✅ 加载完成 ({load_time:.2f}s)")
        except Exception as e:
            print(f"     ❌ BGE-M3加载失败: {e}")
            sys.exit(1)

        print()

    def load_knowledge_base(self):
        """加载知识库"""
        print("📚 正在加载知识库...")

        try:
            with open(self.config['knowledge_base'], 'r', encoding='utf-8') as f:
                content = f.read()

            fragments = [frag.strip() for frag in content.split('。') if frag.strip()]
            self.base_knowledge_fragments = [frag + '。' for frag in fragments]
            self.knowledge_fragments = self.base_knowledge_fragments.copy()

            print(f"     已加载 {len(self.knowledge_fragments)} 个知识片段")
            print("     正在计算向量表示...")

            start_time = time.time()
            self.fragment_embeddings = self.embedder.encode(
                self.knowledge_fragments,
                batch_size=32,
                show_progress_bar=False
            )
            embed_time = time.time() - start_time

            print(f"     ✅ 向量计算完成 ({embed_time:.2f}s)")
            print()

        except Exception as e:
            print(f"     ❌ 知识库加载失败: {e}")
            sys.exit(1)

    def build_faiss_index(self):
        """构建Faiss索引"""
        if FAISS_AVAILABLE:
            print("🎯 构建Faiss索引...")
            start_time = time.time()

            embeddings = self.fragment_embeddings.astype('float32')
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)

            build_time = time.time() - start_time
            print(f"     ✅ Faiss索引构建完成 ({build_time:.2f}s)")
            print(f"     索引向量数: {self.faiss_index.ntotal}")
        else:
            print("⚠️  Faiss不可用，使用基础检索方法")

        print()

    def retrieve_relevant_fragments(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关知识片段"""
        if FAISS_AVAILABLE and self.faiss_index is not None:
            query_embedding = self.embedder.encode([query]).astype('float32')
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            relevant_fragments = [self.knowledge_fragments[i] for i in indices[0]]
            return relevant_fragments
        else:
            query_embedding = self.embedder.encode([query])[0]
            similarities = []
            for frag_embedding in self.fragment_embeddings:
                similarity = np.dot(query_embedding, frag_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(frag_embedding) + 1e-8
                )
                similarities.append(similarity)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_fragments = [self.knowledge_fragments[i] for i in top_indices]
            return relevant_fragments

    def display_system_info(self):
        """显示系统信息"""
        print("=" * 80)
        print("✅ 系统已就绪！")
        print("=" * 80)
        print(f"\n📊 系统信息:")
        print(f"  - LLM: Qwen2-1.5B (MNN)")
        print(f"  - Embedding: BGE-M3")
        print(f"  - 知识库: {len(self.knowledge_fragments)} 个片段")
        print(f"  - 检索方式: {'Faiss' if FAISS_AVAILABLE and self.faiss_index else 'Basic'}")
        print(f"  - 文档状态: {'已加载' if self.has_loaded_documents else '未加载'}")
        print(f"\n💡 提示：输入 'help' 查看所有命令\n")

    def generate_response(self, query: str) -> Tuple[str, List[str]]:
        """生成响应"""
        relevant_fragments = self.retrieve_relevant_fragments(query, top_k=3)
        context = "\n".join([f"- {frag}" for frag in relevant_fragments])
        prompt = f"""根据以下知识库信息回答问题：

【知识库信息】
{context}

【问题】
{query}

【回答】
"""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            response = self.llm_model.response(prompt, stream=False, max_new_tokens=512)

        if hasattr(response, 'text'):
            answer = response.text
        else:
            answer = str(response)

        if "【回答】" in answer:
            answer_start = answer.find("【回答】") + len("【回答】")
            actual_answer = answer[answer_start:].strip()
        else:
            actual_answer = answer

        return actual_answer, relevant_fragments

    def start_chat_session(self):
        """开始聊天会话"""
        print("💬 开始对话")
        print("-" * 80)
        print("输入 'help' 查看命令列表，输入 'quit' 退出")
        print()

        while True:
            try:
                user_input = input("👤 你: ").strip()

                if not user_input:
                    continue

                # 解析命令
                cmd, cmd_type, args = self.command_parser.parse(user_input)

                # 处理非命令输入（普通问题）
                if cmd_type == CommandType.QUERY:
                    print("🤖 正在思考...")
                    start_time = time.time()
                    answer, sources = self.generate_response(user_input)
                    response_time = time.time() - start_time
                    self._display_response(answer, sources, response_time)
                    continue

                # 验证命令
                is_valid, error_msg = self.command_validator.validate(cmd, args)
                if not is_valid:
                    print(error_msg)
                    continue

                # 执行命令
                self._execute_command(cmd, args)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                self._cleanup_on_exit()
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()

    def _execute_command(self, cmd: str, args: list):
        """执行命令"""
        if cmd == 'help':
            print(self.command_parser.get_command_help())

        elif cmd == 'quit':
            print("\n👋 再见！")
            self._cleanup_on_exit()
            sys.exit(0)

        elif cmd == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')

        elif cmd == 'kb':
            self._list_knowledge_base()

        elif cmd == 'cache':
            self._show_document_stats()

        elif cmd == 'doc':
            self._show_document_help()

        elif cmd == 'load':
            file_path = args[0] if args else ''
            self._handle_load_document(file_path)

        elif cmd == 'loaddir':
            self._handle_load_directory()

        elif cmd == 'docs':
            self._show_document_stats()

        else:
            # 提供建议
            suggestion = self.command_validator.suggest_command(cmd)
            if suggestion:
                print(f"❌ 未知命令: {cmd}")
                print(f"💡 你是不是想用: {suggestion}?")
            else:
                print(f"❌ 未知命��: {cmd}")
                print(f"   输入 'help' 查看可用命令")

    def _display_response(self, answer: str, sources: list, response_time: float):
        """显示 LLM 响应"""
        print("\n" + "=" * 80)
        print("💡 回答:")
        print("=" * 80)
        print(answer)
        print("=" * 80)

        print(f"\n📚 参考资料 ({len(sources)} 项):")
        for i, source in enumerate(sources, 1):
            snippet = source[:80] + "..." if len(source) > 80 else source
            print(f"  {i}. {snippet}")

        print(f"\n⏱️  响应时间: {response_time:.2f}s")
        print("-" * 80)

    def _show_document_help(self):
        """显示文档相关命令"""
        print("\n📄 文档相关命令:")
        print("  - 'load <文件路径>': 加载指定文档")
        print("    支持格式: .txt, .md, .docx, .pdf")
        print("    例: load C:\\path\\to\\document.pdf")
        print("  - 'loaddir': 加载文档目录中的所有文档")
        print("  - 'docs'/'cache': 显示已加载文档统计")
        print()

    def _handle_load_document(self, file_path: str):
        """处理单个文档加载"""
        if not file_path:
            print("❌ 请指定文件路径")
            print("   用法: load <文件路径>")
            return

        print(f"\n📥 加载文档: {file_path}")
        success, msg, paragraphs = self.document_manager.load_document(file_path)

        if success:
            print(f"✅ {msg}")
            self.has_loaded_documents = True

            # 重建知识库（包含文档）
            self._rebuild_knowledge_base()
        else:
            print(f"❌ {msg}")

    def _handle_load_directory(self):
        """处理目录加载"""
        doc_dir = self.config['documents_dir']
        print(f"\n📁 加载目录: {doc_dir}")

        results = self.document_manager.load_documents_from_directory(doc_dir)

        if results:
            print(f"\n✅ 加载完成:")
            for doc_name, (success, msg, _) in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {doc_name}: {msg}")

            self.has_loaded_documents = True
            # 重建知识库
            self._rebuild_knowledge_base()
        else:
            print(f"⚠️  目录中未找到支持的文件")

    def _show_document_stats(self):
        """显示文档统计信息"""
        stats = self.document_manager.get_document_stats()

        print("\n📊 文档统计信息:")
        print(f"  - 已加载文档数: {stats['total_documents']}")
        print(f"  - 总段落数: {stats['total_paragraphs']}")
        print(f"  - 总字符数: {stats['total_characters']}")

        if stats['documents']:
            print(f"\n  📄 文档明细:")
            for doc_name, doc_info in stats['documents'].items():
                print(f"    • {doc_name}")
                print(f"      格式: {doc_info['format']}")
                print(f"      段落: {doc_info['paragraphs']}")
                print(f"      字符: {doc_info['characters']}")
        else:
            print("  ⚠️  还未加载任何文档")

        print()

    def _rebuild_knowledge_base(self):
        """使用文档重建知识库"""
        print("\n🔄 重建知识库...")

        # 重置为原始知识库
        self.knowledge_fragments = self.base_knowledge_fragments.copy()

        # 处理文档
        doc_chunks = self.document_manager.process_documents()

        # 合并片段
        if doc_chunks:
            self.knowledge_fragments.extend(doc_chunks)
            print(f"   包含原始知识库片段: {len(self.base_knowledge_fragments)}")
            print(f"   包含文档片段: {len(doc_chunks)}")
            print(f"   总计片段数: {len(self.knowledge_fragments)}")

        print(f"   正在计算向量 ({len(self.knowledge_fragments)} 个片段)...")
        start_time = time.time()

        self.fragment_embeddings = self.embedder.encode(
            self.knowledge_fragments,
            batch_size=32,
            show_progress_bar=False
        )

        embed_time = time.time() - start_time
        print(f"   ✅ 向量计算完成 ({embed_time:.2f}s)")

        # 重建索引
        self.build_faiss_index()
        print("   ✅ 知识库重建完成")

    def _list_knowledge_base(self):
        """列出知识库内容"""
        print(f"\n📚 知识库内容 ({len(self.knowledge_fragments)} 项):")
        display_count = min(20, len(self.knowledge_fragments))
        for i, fragment in enumerate(self.knowledge_fragments[:display_count], 1):
            snippet = fragment[:60] + "..." if len(fragment) > 60 else fragment
            print(f"  {i}. {snippet}")

        if len(self.knowledge_fragments) > display_count:
            print(f"  ... 还有 {len(self.knowledge_fragments) - display_count} 项")
        print()

    def _cleanup_on_exit(self):
        """
        ⭐ 系统退出时的清理逻辑
        关键：清除文档产生的向量，恢复到原始知识库
        """
        print("\n🧹 正在清理...")

        if self.has_loaded_documents:
            print("   清除文档产生的向量存储...")
            self.document_manager.clear_documents()
            self.has_loaded_documents = False

            # 恢复原始知识库
            self.knowledge_fragments = self.base_knowledge_fragments.copy()
            self.fragment_embeddings = self.embedder.encode(
                self.knowledge_fragments,
                batch_size=32,
                show_progress_bar=False
            )

            # 重建索引
            self.build_faiss_index()
            print("   ✅ 已恢复原始知识库")

        print("   ✅ 清理完成")


def main():
    """主函数"""
    try:
        rag_system = RAGSystem()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()