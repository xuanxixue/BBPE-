import sys
import os
import logging
from typing import List, Dict, Any, Optional
import threading
from datetime import datetime
import tempfile
import json
import time
import regex as re
from collections import Counter

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QListWidget, QLabel, 
                             QWidget, QFileDialog, QSplitter, QProgressBar,
                             QMessageBox, QTabWidget, QFrame, QLineEdit, 
                             QGroupBox, QCheckBox, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor

# 分词器相关导入
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
    from tokenizers.processors import TemplateProcessing
    import docx
    TOKENIZERS_AVAILABLE = True
except ImportError as e:
    print(f"导入错误: {e}")
    TOKENIZERS_AVAILABLE = False

class TrainingSignals(QObject):
    """训练信号类，用于线程间通信"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    training_complete = pyqtSignal(bool, str)
    tokenizer_info = pyqtSignal(dict)

class ChineseAwareProcessor:
    """中文感知处理器 - 专门优化中文处理"""
    
    @staticmethod
    def preprocess_chinese_text(text: str) -> str:
        """预处理中文文本，保留中文词汇完整性"""
        # 中文正则表达式：匹配中文字符
        chinese_pattern = r'([\u4e00-\u9fff]+)'
        
        # 在中文词汇之间添加空格，但保持中文词汇内部不分割
        processed = re.sub(chinese_pattern, r' \1 ', text)
        
        # 清理多余空格
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    @staticmethod
    def extract_chinese_words(text: str) -> List[str]:
        """提取中文词汇"""
        chinese_pattern = r'[\u4e00-\u9fff]+'
        return re.findall(chinese_pattern, text)
    
    @staticmethod
    def calculate_chinese_frequency(texts: List[str]) -> Dict[str, int]:
        """计算中文词汇频率"""
        all_chinese_words = []
        for text in texts:
            words = ChineseAwareProcessor.extract_chinese_words(text)
            all_chinese_words.extend(words)
        
        return Counter(all_chinese_words)

class DocumentProcessor:
    """文档处理器 - 优化中文处理"""
    
    def __init__(self):
        self.chinese_processor = ChineseAwareProcessor()
    
    @staticmethod
    def read_txt_file(file_path: str) -> str:
        """读取txt文件"""
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        # 过滤掉空行和过短的内容
                        lines = [line.strip() for line in content.split('\n') if len(line.strip()) > 5]
                        return '\n'.join(lines)
                except UnicodeDecodeError:
                    continue
            raise Exception(f"无法解码文件: {file_path}")
        except Exception as e:
            raise Exception(f"读取txt文件失败: {e}")
    
    @staticmethod
    def read_docx_file(file_path: str) -> str:
        """读取docx文件"""
        try:
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip() and len(paragraph.text.strip()) > 3:
                    text.append(paragraph.text.strip())
            return '\n'.join(text)
        except Exception as e:
            raise Exception(f"读取docx文件失败: {e}")
    
    def extract_text_from_files(self, file_paths: List[str], preprocess_chinese: bool = True) -> List[str]:
        """从多个文件中提取文本"""
        texts = []
        total_size = 0
        for file_path in file_paths:
            try:
                if file_path.lower().endswith('.txt'):
                    text = self.read_txt_file(file_path)
                elif file_path.lower().endswith('.docx'):
                    text = self.read_docx_file(file_path)
                else:
                    continue
                
                if text.strip():
                    # 中文预处理
                    if preprocess_chinese:
                        text = self.chinese_processor.preprocess_chinese_text(text)
                    texts.append(text)
                    total_size += len(text.encode('utf-8'))
            except Exception as e:
                logging.warning(f"读取文件 {file_path} 失败: {e}")
        
        return texts, total_size

class HybridBPEChineseTrainer:
    """混合BPE中文分词器训练器 - 结合传统BPE和中文优化"""
    
    def __init__(self, signals: TrainingSignals):
        self.signals = signals
        self.tokenizer = None
        self.is_trained = False
        self.vocab_size = 0
        self.document_processor = DocumentProcessor()
    
    def create_temp_training_files(self, texts: List[str]) -> List[str]:
        """创建临时训练文件"""
        temp_files = []
        
        # 将所有文本合并
        all_text = '\n'.join(texts)
        
        # 分块写入文件
        chunk_size = 500000  # 500KB每块
        for i in range(0, len(all_text), chunk_size):
            chunk = all_text[i:i + chunk_size]
            temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                                  suffix=f'_chunk_{i//chunk_size}.txt', delete=False)
            with temp_file:
                temp_file.write(chunk)
            temp_files.append(temp_file.name)
        
        return temp_files
    
    def train_tokenizer(self, file_paths: List[str], config: Dict[str, Any]):
        """训练混合BPE分词器 - 专门优化中文"""
        try:
            self.signals.log_signal.emit("开始初始化混合BPE中文分词器...")
            self.signals.log_signal.emit("正在读取训练文件...")
            
            start_time = time.time()
            
            # 提取文本并进行中文预处理
            preprocess_chinese = config.get('preprocess_chinese', True)
            texts, total_size = self.document_processor.extract_text_from_files(file_paths, preprocess_chinese)
            if not texts:
                raise Exception("没有找到可用的文本内容")
            
            read_time = time.time() - start_time
            self.signals.log_signal.emit(f"成功提取 {len(texts)} 个文档，总大小: {total_size/1024/1024:.2f} MB")
            self.signals.log_signal.emit(f"文件读取耗时: {read_time:.2f} 秒")
            
            # 分析中文词汇频率
            if preprocess_chinese:
                chinese_freq = ChineseAwareProcessor.calculate_chinese_frequency(texts)
                self.signals.log_signal.emit(f"发现 {len(chinese_freq)} 个不同的中文词汇")
                if chinese_freq:
                    top_words = list(chinese_freq.items())[:10]
                    self.signals.log_signal.emit(f"高频中文词汇示例: {', '.join([f'{word}({count})' for word, count in top_words])}")
            
            if total_size < 1024 * 1024:
                self.signals.log_signal.emit("警告: 训练数据较小，建议提供更多数据以获得更好的分词效果")
            
            # 创建临时训练文件
            self.signals.log_signal.emit("正在准备训练数据...")
            temp_files = self.create_temp_training_files(texts)
            
            vocab_size = config.get('vocab_size', 40000)
            self.vocab_size = vocab_size
            
            # 训练分词器
            self._train_hybrid_bpe(temp_files, config, total_size)
            
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            total_time = time.time() - start_time
            self.is_trained = True
            self.signals.log_signal.emit(f"混合BPE中文分词器训练完成！总耗时: {total_time:.2f} 秒")
            
            # 发送分词器信息
            info = {
                'vocab_size': self.vocab_size,
                'model_type': 'hybrid_bpe',
                'file_count': len(file_paths),
                'data_size_mb': total_size/1024/1024,
                'training_time': total_time
            }
            self.signals.tokenizer_info.emit(info)
            self.signals.training_complete.emit(True, "训练成功完成")
            
        except Exception as e:
            error_msg = f"训练过程中出现错误: {str(e)}"
            self.signals.log_signal.emit(error_msg)
            self.signals.training_complete.emit(False, error_msg)
    
    def _train_hybrid_bpe(self, files: List[str], config: Dict[str, Any], data_size: int):
        """训练混合BPE分词器"""
        self.signals.log_signal.emit("初始化混合BPE分词器模型...")
        
        # 创建tokenizer - 使用传统BPE而不是BBPE
        self.tokenizer = Tokenizer(BPE(
            unk_token=config.get('unk_token', '[UNK]')
        ))
        
        # 使用空格和标点预分词器 - 更适合中文
        pre_tokenizers = []
        if config.get('use_whitespace', True):
            pre_tokenizers.append(Whitespace())
        if config.get('use_punctuation', True):
            pre_tokenizers.append(Punctuation())
        
        if pre_tokenizers:
            if len(pre_tokenizers) == 1:
                self.tokenizer.pre_tokenizer = pre_tokenizers[0]
            else:
                self.tokenizer.pre_tokenizer = Sequence(pre_tokenizers)
        
        # 配置特殊token
        special_tokens = config.get('special_tokens', [
            "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"
        ])
        
        self.signals.log_signal.emit(f"开始混合BPE训练，词汇表大小: {config.get('vocab_size', 40000)}")
        self.signals.log_signal.emit(f"训练文件数量: {len(files)}，数据量: {data_size/1024/1024:.2f} MB")
        
        # 配置训练器
        trainer = BpeTrainer(
            vocab_size=config.get('vocab_size', 40000),
            min_frequency=config.get('min_frequency', 2),
            special_tokens=special_tokens,
            show_progress=True,
            # 不使用字节级别配置，使用传统的BPE配置
            initial_alphabet=[],
        )
        
        # 实际训练
        train_start = time.time()
        self.tokenizer.train(files, trainer=trainer)
        train_time = time.time() - train_start
        
        self.signals.log_signal.emit(f"混合BPE训练完成，耗时: {train_time:.2f} 秒")
        
        # 配置后处理
        self._setup_post_processing(config)
        
        # 验证分词器
        vocab_size_actual = self.tokenizer.get_vocab_size()
        self.signals.log_signal.emit(f"实际词汇表大小: {vocab_size_actual}")
        
        # 测试分词器
        test_text = "三月七喜欢帕姆。Hello world! 这是测试。"
        test_encoding = self.tokenizer.encode(test_text)
        self.signals.log_signal.emit(f"测试分词: '{test_text}' -> {len(test_encoding.tokens)} tokens")
        self.signals.log_signal.emit(f"测试tokens: {test_encoding.tokens}")
    
    def _setup_post_processing(self, config: Dict[str, Any]):
        """配置后处理"""
        if config.get('add_special_tokens', True):
            try:
                self.tokenizer.post_processor = TemplateProcessing(
                    single=config.get('template_single', "[CLS] $A [SEP]"),
                    pair=config.get('template_pair', "[CLS] $A [SEP] $B:1 [SEP]:1"),
                    special_tokens=[
                        ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                        ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
                    ],
                )
            except Exception as e:
                self.signals.log_signal.emit(f"后处理配置警告: {e}")
    
    def tokenize_text(self, text: str, preprocess: bool = True) -> Dict[str, Any]:
        """对文本进行分词"""
        if not self.is_trained or self.tokenizer is None:
            raise Exception("分词器尚未训练")
        
        try:
            # 预处理文本
            if preprocess:
                processed_text = ChineseAwareProcessor.preprocess_chinese_text(text)
            else:
                processed_text = text
            
            encoding = self.tokenizer.encode(processed_text)
            
            # 清理token显示
            clean_tokens = []
            for token in encoding.tokens:
                # 清理特殊字符显示
                clean_token = token.replace('Ġ', ' ').strip()
                if clean_token in ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']:
                    clean_tokens.append(clean_token)
                elif clean_token:
                    clean_tokens.append(clean_token)
            
            # 获取解码后的文本
            decoded_text = self.tokenizer.decode(encoding.ids)
            
            # 分析token长度
            token_lengths = [len(token.replace('Ġ', '')) for token in encoding.tokens if token not in ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']]
            
            return {
                'tokens': encoding.tokens,
                'clean_tokens': clean_tokens,
                'ids': encoding.ids,
                'token_count': len(encoding.tokens),
                'original_text': text,
                'processed_text': processed_text,
                'decoded_text': decoded_text,
                'token_lengths': token_lengths,
                'attention_mask': encoding.attention_mask,
                'type_ids': encoding.type_ids if hasattr(encoding, 'type_ids') else None
            }
        except Exception as e:
            raise Exception(f"分词失败: {e}")
    
    def save_tokenizer(self, path: str):
        """保存分词器"""
        if self.tokenizer:
            self.tokenizer.save(path)
    
    def load_tokenizer(self, path: str):
        """加载分词器"""
        self.tokenizer = Tokenizer.from_file(path)
        self.is_trained = True

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.uploaded_files = []
        self.tokenizer_trainer = None
        self.training_thread = None
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("混合BPE中文分词器训练系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("混合BPE中文分词器训练系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        main_layout.addWidget(title_label)
        
        # 创建标签页
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 训练标签页
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        
        # 文件上传区域
        upload_frame = self.create_upload_frame()
        train_layout.addWidget(upload_frame)
        
        # 配置区域
        config_frame = self.create_config_frame()
        train_layout.addWidget(config_frame)
        
        # 训练控制区域
        training_frame = self.create_training_frame()
        train_layout.addWidget(training_frame)
        
        # 日志区域
        log_frame = self.create_log_frame()
        train_layout.addWidget(log_frame)
        
        # 测试标签页
        test_tab = QWidget()
        test_layout = QVBoxLayout(test_tab)
        test_frame = self.create_test_frame()
        test_layout.addWidget(test_frame)
        
        tab_widget.addTab(train_tab, "训练")
        tab_widget.addTab(test_tab, "测试")
        
    def create_upload_frame(self) -> QFrame:
        """创建文件上传区域"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(frame)
        
        # 标题
        upload_label = QLabel("文档上传区域 (特别优化中文文本)")
        upload_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(upload_label)
        
        # 上传按钮
        upload_layout = QHBoxLayout()
        upload_btn = QPushButton("上传TXT文档")
        upload_btn.clicked.connect(lambda: self.upload_files('txt'))
        upload_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 12px; }")
        upload_layout.addWidget(upload_btn)
        
        upload_docx_btn = QPushButton("上传DOCX文档")
        upload_docx_btn.clicked.connect(lambda: self.upload_files('docx'))
        upload_docx_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 12px; }")
        upload_layout.addWidget(upload_docx_btn)
        
        clear_btn = QPushButton("清空文件列表")
        clear_btn.clicked.connect(self.clear_files)
        clear_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 12px; background-color: #ff6b6b; color: white; }")
        upload_layout.addWidget(clear_btn)
        
        layout.addLayout(upload_layout)
        
        # 文件列表和统计
        file_stats_layout = QHBoxLayout()
        
        # 文件列表
        file_list_frame = QFrame()
        file_list_layout = QVBoxLayout(file_list_frame)
        file_list_layout.addWidget(QLabel("已上传文件:"))
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        file_list_layout.addWidget(self.file_list)
        
        # 统计信息
        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.addWidget(QLabel("统计信息:"))
        self.stats_label = QLabel("文件数量: 0 | 总大小: 0 MB")
        self.stats_label.setStyleSheet("background-color: #f8f9fa; padding: 5px; border: 1px solid #dee2e6;")
        stats_layout.addWidget(self.stats_label)
        
        file_stats_layout.addWidget(file_list_frame, 70)
        file_stats_layout.addWidget(stats_frame, 30)
        layout.addLayout(file_stats_layout)
        
        return frame
    
    def create_config_frame(self) -> QFrame:
        """创建配置区域"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(frame)
        
        # 标题
        config_label = QLabel("混合BPE中文分词器配置")
        config_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(config_label)
        
        # 基本配置
        basic_group = QGroupBox("BPE训练参数")
        basic_layout = QHBoxLayout(basic_group)
        
        basic_layout.addWidget(QLabel("词汇表大小:"))
        self.vocab_size_edit = QSpinBox()
        self.vocab_size_edit.setRange(5000, 100000)
        self.vocab_size_edit.setValue(40000)
        self.vocab_size_edit.setSingleStep(5000)
        basic_layout.addWidget(self.vocab_size_edit)
        
        basic_layout.addWidget(QLabel("最小频率:"))
        self.min_freq_edit = QSpinBox()
        self.min_freq_edit.setRange(1, 100)
        self.min_freq_edit.setValue(2)
        basic_layout.addWidget(self.min_freq_edit)
        
        basic_layout.addStretch()
        layout.addWidget(basic_group)
        
        # 中文优化配置
        chinese_group = QGroupBox("中文优化配置")
        chinese_layout = QVBoxLayout(chinese_group)
        
        chinese_options_layout = QHBoxLayout()
        self.preprocess_chinese_check = QCheckBox("中文预处理")
        self.preprocess_chinese_check.setChecked(True)
        chinese_options_layout.addWidget(self.preprocess_chinese_check)
        
        self.use_whitespace_check = QCheckBox("使用空格分割")
        self.use_whitespace_check.setChecked(True)
        chinese_options_layout.addWidget(self.use_whitespace_check)
        
        self.use_punctuation_check = QCheckBox("使用标点分割")
        self.use_punctuation_check.setChecked(True)
        chinese_options_layout.addWidget(self.use_punctuation_check)
        
        chinese_options_layout.addStretch()
        chinese_layout.addLayout(chinese_options_layout)
        
        chinese_note = QLabel("中文预处理会在中文词汇之间添加空格，保持词汇完整性，显著改善中文分词效果")
        chinese_note.setStyleSheet("color: #666; font-size: 10px;")
        chinese_layout.addWidget(chinese_note)
        
        layout.addWidget(chinese_group)
        
        # 特殊token配置
        special_group = QGroupBox("特殊Token配置")
        special_layout = QHBoxLayout(special_group)
        
        self.add_special_check = QCheckBox("添加特殊token")
        self.add_special_check.setChecked(True)
        special_layout.addWidget(self.add_special_check)
        
        special_note = QLabel("使用BERT风格的特殊token: [CLS], [SEP], [UNK], [PAD], [MASK]")
        special_note.setStyleSheet("color: #666; font-size: 10px;")
        special_layout.addWidget(special_note)
        
        special_layout.addStretch()
        layout.addWidget(special_group)
        
        # 技术说明
        info_group = QGroupBox("技术说明")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "混合BPE中文分词器结合了传统BPE算法和中文优化处理：\n"
            "• 使用字符级BPE而不是字节级BBPE\n"
            "• 专门优化中文词汇处理\n"
            "• 保持中文词汇完整性\n"
            "• 支持多语言混合文本\n"
            "• 训练效果优于纯BBPE方法"
        )
        info_text.setStyleSheet("color: #444; font-size: 10px; padding: 5px;")
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        return frame
    
    def create_training_frame(self) -> QFrame:
        """创建训练控制区域"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(frame)
        
        # 训练按钮
        self.train_btn = QPushButton("开始训练混合BPE分词器")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet("""
            QPushButton { 
                padding: 12px; 
                font-size: 14px; 
                font-weight: bold; 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 5px;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
            QPushButton:disabled { 
                background-color: #cccccc; 
            }
        """)
        layout.addWidget(self.train_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 分词器信息
        self.tokenizer_info_label = QLabel("分词器状态: 未训练")
        self.tokenizer_info_label.setStyleSheet("padding: 5px; background-color: #e9e9e9;")
        layout.addWidget(self.tokenizer_info_label)
        
        return frame
    
    def create_log_frame(self) -> QFrame:
        """创建日志显示区域"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(frame)
        
        # 标题和清空按钮
        log_header_layout = QHBoxLayout()
        log_label = QLabel("训练日志")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        log_header_layout.addWidget(log_label)
        
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("QPushButton { padding: 4px; font-size: 10px; }")
        log_header_layout.addStretch()
        log_header_layout.addWidget(clear_log_btn)
        
        layout.addLayout(log_header_layout)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(250)
        self.log_text.setStyleSheet("font-family: 'Courier New'; font-size: 10pt;")
        layout.addWidget(self.log_text)
        
        return frame
    
    def create_test_frame(self) -> QFrame:
        """创建测试区域"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout(frame)
        
        # 标题
        test_label = QLabel("分词测试区域")
        test_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(test_label)
        
        # 测试输入
        layout.addWidget(QLabel("输入测试文本:"))
        self.test_input = QTextEdit()
        self.test_input.setMaximumHeight(100)
        self.test_input.setPlaceholderText("在此输入要分词的文本...\n例如: 三月七喜欢帕姆。")
        layout.addWidget(self.test_input)
        
        # 测试选项
        test_options_layout = QHBoxLayout()
        test_options_layout.addWidget(QLabel("中文预处理:"))
        self.test_preprocess_check = QCheckBox()
        self.test_preprocess_check.setChecked(True)
        test_options_layout.addWidget(self.test_preprocess_check)
        test_options_layout.addStretch()
        layout.addLayout(test_options_layout)
        
        # 测试按钮
        test_btn = QPushButton("测试分词")
        test_btn.clicked.connect(self.test_tokenization)
        test_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 12px; background-color: #2196F3; color: white; }")
        layout.addWidget(test_btn)
        
        # 测试结果
        layout.addWidget(QLabel("分词结果:"))
        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setStyleSheet("font-family: 'Courier New'; font-size: 10pt;")
        layout.addWidget(self.test_output)
        
        return frame
    
    def setup_logging(self):
        """设置日志系统"""
        self.training_signals = TrainingSignals()
        self.training_signals.log_signal.connect(self.update_log)
        self.training_signals.progress_signal.connect(self.update_progress)
        self.training_signals.training_complete.connect(self.training_finished)
        self.training_signals.tokenizer_info.connect(self.update_tokenizer_info)
    
    def upload_files(self, file_type: str):
        """上传文件"""
        if file_type == 'txt':
            filter_str = "文本文件 (*.txt);;所有文件 (*.*)"
        else:
            filter_str = "Word文档 (*.docx);;所有文件 (*.*)"
        
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            f"选择{file_type.upper()}文档文件", 
            "", 
            filter_str
        )
        
        if files:
            for file_path in files:
                if file_path not in self.uploaded_files:
                    self.uploaded_files.append(file_path)
                    file_name = os.path.basename(file_path)
                    self.file_list.addItem(file_name)
            
            # 更新统计信息
            total_size = sum(os.path.getsize(f) for f in self.uploaded_files)
            self.stats_label.setText(f"文件数量: {len(self.uploaded_files)} | 总大小: {total_size/1024/1024:.2f} MB")
            
            self.update_log(f"成功上传 {len(files)} 个{file_type.upper()}文件，总大小: {total_size/1024/1024:.2f} MB")
    
    def clear_files(self):
        """清空文件列表"""
        self.uploaded_files.clear()
        self.file_list.clear()
        self.stats_label.setText("文件数量: 0 | 总大小: 0 MB")
        self.update_log("已清空文件列表")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return {
            'vocab_size': self.vocab_size_edit.value(),
            'min_frequency': self.min_freq_edit.value(),
            'preprocess_chinese': self.preprocess_chinese_check.isChecked(),
            'use_whitespace': self.use_whitespace_check.isChecked(),
            'use_punctuation': self.use_punctuation_check.isChecked(),
            'add_special_tokens': self.add_special_check.isChecked(),
            'special_tokens': [
                "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"
            ],
            'unk_token': "[UNK]",
            'template_single': "[CLS] $A [SEP]",
            'template_pair': "[CLS] $A [SEP] $B:1 [SEP]:1"
        }
    
    def start_training(self):
        """开始训练"""
        if not self.uploaded_files:
            QMessageBox.warning(self, "警告", "请先上传训练文件")
            return
        
        if not TOKENIZERS_AVAILABLE:
            QMessageBox.critical(self, "错误", 
                               "分词器库未正确安装。请运行: pip install tokenizers python-docx regex")
            return
        
        # 检查数据量
        total_size = sum(os.path.getsize(f) for f in self.uploaded_files)
        if total_size < 1024 * 1024:
            reply = QMessageBox.question(self, "数据量较小", 
                                       f"当前训练数据只有 {total_size/1024/1024:.2f} MB，可能影响训练效果。是否继续？",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # 禁用训练按钮
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # 清空日志
        self.log_text.clear()
        
        # 获取训练配置
        config = self.get_training_config()
        
        # 在后台线程中训练
        self.tokenizer_trainer = HybridBPEChineseTrainer(self.training_signals)
        self.training_thread = threading.Thread(
            target=self.tokenizer_trainer.train_tokenizer,
            args=(self.uploaded_files, config)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def test_tokenization(self):
        """测试分词"""
        if not self.tokenizer_trainer or not self.tokenizer_trainer.is_trained:
            QMessageBox.warning(self, "警告", "请先训练分词器")
            return
        
        text = self.test_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请输入测试文本")
            return
        
        try:
            use_preprocess = self.test_preprocess_check.isChecked()
            result = self.tokenizer_trainer.tokenize_text(text, use_preprocess)
            
            # 格式化输出结果 - 更清晰的显示
            output = f"输入文本：{result['original_text']}\n\n"
            
            output += f"分词结果（{len(result['clean_tokens'])}个token）：\n"
            # 过滤掉特殊token，只显示实际内容token
            content_tokens = [token for token in result['clean_tokens'] if token not in ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']]
            output += " | ".join(content_tokens) + "\n\n"
            
            output += "Token列表：\n"
            for i, token in enumerate(content_tokens, 1):
                length = result['token_lengths'][i-1] if i-1 < len(result['token_lengths']) else len(token)
                output += f"{i}. '{token}'（长度：{length}）\n"
            
            output += f"\nToken IDs: {result['ids']}\n"
            
            self.test_output.setPlainText(output)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分词测试失败: {str(e)}")
    
    def update_log(self, message: str):
        """更新日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_text.append(log_entry)
        
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_tokenizer_info(self, info: Dict[str, Any]):
        """更新分词器信息"""
        model_type = info.get('model_type', 'unknown')
        vocab_size = info.get('vocab_size', 0)
        file_count = info.get('file_count', 0)
        data_size = info.get('data_size_mb', 0)
        training_time = info.get('training_time', 0)
        
        status_text = f"分词器状态: 已训练 | 模型: {model_type} | 词汇表: {vocab_size} | 训练文件: {file_count} | 数据量: {data_size:.2f}MB | 耗时: {training_time:.1f}s"
        self.tokenizer_info_label.setText(status_text)
    
    def training_finished(self, success: bool, message: str):
        """训练完成回调"""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "成功", "混合BPE分词器训练完成！")
        else:
            QMessageBox.critical(self, "错误", f"训练失败: {message}")

def check_dependencies():
    """检查依赖是否安装"""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    if not TOKENIZERS_AVAILABLE:
        missing_deps.append("tokenizers")
    
    try:
        import docx
    except ImportError:
        missing_deps.append("python-docx")
    
    try:
        import regex
    except ImportError:
        missing_deps.append("regex")
    
    return missing_deps

def main():
    """主函数"""
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        print("缺少必要的依赖包:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请使用以下命令安装:")
        print("pip install PyQt5 tokenizers python-docx regex")
        return
    
    app = QApplication(sys.argv)
    app.setApplicationName("混合BPE中文分词器训练系统")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()