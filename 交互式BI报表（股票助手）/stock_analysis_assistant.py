import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine, text
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
from tavily import TavilyClient
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入ARIMA和MACD所需的库
from statsmodels.tsa.arima.model import ARIMA

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配置 DashScope（通义千问 API）
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
if not dashscope.api_key:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量以使用通义千问 API")
dashscope.timeout = 30

# 配置 Tavily（新闻搜索 API）
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# 初始化Tavily客户端
def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

# 获取MySQL连接配置
def get_mysql_config():
    """从环境变量获取并返回MySQL连接配置"""
    mysql_user = os.getenv('MYSQL_USER', 'root')
    mysql_password = os.getenv('MYSQL_PASSWORD')
    mysql_host = os.getenv('MYSQL_HOST', 'localhost')
    mysql_port = os.getenv('MYSQL_PORT', '3306')
    
    if not mysql_password:
        raise ValueError("请设置 MYSQL_PASSWORD 环境变量以连接数据库")
    
    return {
        'user': mysql_user,
        'password': mysql_password,
        'host': mysql_host,
        'port': mysql_port
    }

# ====== 股票助手 system prompt 和函数描述 ======
system_prompt = '''我是股票查询与分析助手，以下是关于股票历史交易数据表相关的字段，我可以编写对应的SQL查询语句，对股票数据进行查询、分析和可视化，还可以使用ARIMA模型进行价格预测以及使用MACD指标进行交易分析
-- 股票历史交易数据表
CREATE TABLE `all_symbols` (
    `id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID，自增',
    `stock_code` VARCHAR(20) NOT NULL COMMENT '股票代码，格式如：600519.SH, 000858.SZ',
    `stock_name` VARCHAR(50) NOT NULL COMMENT '股票名称，英文名称',
    `trade_date` VARCHAR(10) NOT NULL COMMENT '交易日期，格式如：YYYYMMDD 或 YYYY-MM-DD',
    `open` DECIMAL(10, 2) NOT NULL COMMENT '开盘价',
    `high` DECIMAL(10, 2) NOT NULL COMMENT '最高价',
    `low` DECIMAL(10, 2) NOT NULL COMMENT '最低价',
    `close` DECIMAL(10, 2) NOT NULL COMMENT '收盘价',
    `pre_close` DECIMAL(10, 2) NOT NULL COMMENT '前一日收盘价',
    `change` DECIMAL(10, 2) NOT NULL COMMENT '涨跌额',
    `pct_chg` DECIMAL(10, 2) NOT NULL COMMENT '涨跌幅（%）',
    `volume` BIGINT NOT NULL COMMENT '成交量（手）',
    `amount` DECIMAL(15, 2) NOT NULL COMMENT '成交额（万元）'
);

常用股票代码：
- 贵州茅台: 600519.SH
- 中芯国际: 688981.SH
- 国泰君安: 601211.SH
- 五粮液: 000858.SZ

主要功能：
1. 股票基本信息查询（开盘价、收盘价、最高价、最低价等）
2. 股票价格趋势分析
3. 成交量和成交额分析
4. 涨跌幅计算和统计
5. 多股票对比分析
6. 最新股票相关新闻查询（通过 Tavily 搜索引擎获取）
7. 使用ARIMA模型进行未来N天股票价格预测
8. 使用MACD指标进行交易策略分析，包括过去一年的买卖点和收益率计算

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不能省略图片。这样用户才能直接看到表格和图片。
''' 
functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询，支持股票数据的查询、分析和可视化",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
    {
        "name": "get_stock_news",
        "description": "获取指定股票的最新相关新闻，支持按股票名称或代码查询",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_name_or_code": {
                    "type": "string",
                    "description": "股票名称或代码，例如：贵州茅台、600519.SH、五粮液等",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回的新闻数量，默认为5条",
                    "default": 5
                }
            },
            "required": ["stock_name_or_code"],
        },
    },
    {
        "name": "arima_stock",
        "description": "使用ARIMA模型预测股票未来N天的价格，从MySQL获取历史数据进行建模",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，例如：600519.SH、000858.SZ等",
                },
                "n": {
                    "type": "integer",
                    "description": "预测未来的天数，默认为7天",
                    "default": 7
                }
            },
            "required": ["ts_code"],
        },
    },
    {
        "name": "macd_stock",
        "description": "使用MACD指标分析股票交易策略，从MySQL获取历史数据，计算买卖点和收益率",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，例如：600519.SH、000858.SZ等",
                }
            },
            "required": ["ts_code"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
_last_df_dict = {}

def get_session_id(kwargs):
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== Tavily 新闻搜索工具类 ======
@register_tool('get_stock_news')
class GetStockNewsTool(BaseTool):
    description = '获取指定股票的最新相关新闻，支持按股票名称或代码查询'
    parameters = [
        {
            'name': 'stock_name_or_code',
            'type': 'string',
            'description': '股票名称或代码，例如：贵州茅台、600519.SH、五粮液等',
            'required': True
        },
        {
            'name': 'max_results',
            'type': 'integer',
            'description': '返回的新闻数量，默认为5条',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        stock_name_or_code = args['stock_name_or_code']
        max_results = args.get('max_results', 5)
        
        # 检查Tavily API Key是否配置
        if not TAVILY_API_KEY:
            return "错误：请设置 TAVILY_API_KEY 环境变量以使用新闻搜索功能"
            
        try:
            client = get_tavily_client()
            search_query = f"{stock_name_or_code} 股票 最新新闻"
            
            response = client.search(
                query=search_query,
                search_depth='advanced',
                max_results=max_results
            )
            
            if 'results' in response:
                news_items = response['results']
                news_md = f"### {stock_name_or_code} 最新相关新闻\n\n"
                
                for idx, item in enumerate(news_items, 1):
                    news_md += f"**{idx}. {item.get('title', '无标题')}**\n"
                    news_md += f"{item.get('content', '无内容摘要')}\n"
                    news_md += f"来源：{item.get('url', '未知来源')}\n\n"
                
                return news_md
            else:
                return f"未找到关于 {stock_name_or_code} 的相关新闻。"
                
        except Exception as e:
            return f"获取新闻时发生错误：{str(e)}"

# ====== RAG 知识库工具类 ======
@register_tool('rag_stock_knowledge')
class RAGStockKnowledgeTool(BaseTool):
    description = '股票知识库检索工具，用于回答股票相关的基础知识、技术分析方法和投资策略问题，从本地FAQ知识库获取信息'
    parameters = [
        {
            'name': 'query',
            'type': 'string',
            'description': '用户的问题，例如：如何对比两只股票的涨跌幅、MACD指标是什么等',
            'required': True
        }
    ]

    def _load_knowledge_base(self):
        knowledge_path = './faq.txt'
        if not os.path.exists(knowledge_path):
            return []
        
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析FAQ格式：Q1: ... A1: ...
            import re
            # 匹配Qn和对应的An
            pattern = r'Q(\d+)：(.+?)\nA(\d+)：(.+?)(?=\nQ\d+:|$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            knowledge_base = []
            for match in matches:
                q_num, question, a_num, answer = match
                knowledge_base.append({
                    'question': question.strip(),
                    'answer': answer.strip(),
                    'keywords': self._extract_keywords(question + ' ' + answer)
                })
            
            return knowledge_base
        except Exception as e:
            print(f"加载知识库时发生错误：{str(e)}")
            return []
    
    def _extract_keywords(self, text):
        # 简单的关键词提取：去除停用词，保留主要词汇
        stop_words = {'的', '了', '和', '是', '在', '我', '有', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 简单分词（按空格和标点符号）
        import re
        words = re.findall(r'\w+', text)
        
        # 过滤停用词，保留长度大于1的词
        keywords = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 1]
        
        return list(set(keywords))  # 去重
    
    def _search_knowledge(self, query):
        query_keywords = self._extract_keywords(query)
        
        # 加载知识库
        knowledge_base = self._load_knowledge_base()
        
        # 计算每个知识库条目与查询的相似度（基于关键词匹配）
        results = []
        for item in knowledge_base:
            # 计算关键词匹配数
            match_count = len(set(query_keywords) & set(item['keywords']))
            
            if match_count > 0:
                results.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'relevance': match_count
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # 返回前3个最相关的结果
        return results[:3]
    
    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        query = args['query']
        
        try:
            # 在知识库中检索
            results = self._search_knowledge(query)
            
            if results:
                # 格式化检索结果
                response = f"### 从股票知识库中检索到的相关信息\n\n"
                
                for i, result in enumerate(results, 1):
                    response += f"**{i}. 相关问题：** {result['question']}\n"
                    response += f"**回答：** {result['answer']}\n\n"
                
                return response
            else:
                return f"知识库中未找到与 '{query}' 相关的信息。"
                
        except Exception as e:
            return f"知识库检索时发生错误：{str(e)}"

# ====== MACD 股票交易分析工具类 ======
@register_tool('macd_stock')
class MacdStockTool(BaseTool):
    """
    使用MACD指标分析股票交易策略，包括买卖点确定和收益率计算
    """
    description = '使用MACD指标分析股票交易策略，从MySQL获取历史数据，计算买卖点和收益率'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，例如：600519.SH、000858.SZ等',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import os, time
        args = json.loads(params)
        ts_code = args['ts_code']
        
        try:
            # 从MySQL获取历史数据
            df = self.get_historical_data(ts_code)
            if df.empty:
                return f"未找到股票代码 {ts_code} 的历史数据"
            
            # 计算MACD指标
            df = self.calculate_macd(df)
            
            # 确定买卖点
            trade_signals = self.generate_trade_signals(df)
            
            # 计算收益率
            performance = self.calculate_performance(df, trade_signals)
            
            # 生成可视化图表
            save_dir = os.path.join(os.path.dirname(__file__), 'stock_charts')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'macd_strategy_{ts_code}_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            img_path = os.path.join('stock_charts', filename)
            
            # 生成MACD策略图表
            self.plot_macd_strategy(df, trade_signals, save_path, ts_code)
            
            # 准备返回结果
            result_md = self.format_result(ts_code, trade_signals, performance)
            result_md += f"\n![{ts_code} MACD交易策略]({img_path})"
            
            return result_md
            
        except Exception as e:
            error_msg = f"MACD交易分析失败: {str(e)}"
            print(error_msg)
            return error_msg

    def get_historical_data(self, ts_code):
        """从MySQL获取股票历史数据"""
        # 从环境变量获取MySQL连接信息
        mysql_config = get_mysql_config()
        database = 'stock_data'
        
        try:
            engine = create_engine(
                f'mysql+pymysql://{mysql_config["user"]}:{mysql_config["password"]}@{mysql_config["host"]}:{mysql_config["port"]}/{database}?charset=utf8mb4',
                connect_args={'connect_timeout': 10}
            )
            
            # 计算日期范围：截止到今天的前一年
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 查询股票数据
            sql = text("""
                SELECT trade_date, close FROM all_symbols 
                WHERE stock_code = :code 
                AND trade_date BETWEEN :start AND :end
                ORDER BY trade_date
            """)
            
            df = pd.read_sql(sql, engine, params={'code': ts_code, 'start': start_date, 'end': end_date})
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            # 移除 asfreq 转换，避免索引问题
            # df = df.asfreq('B')  # 使用工作日频率
            # df = df.fillna(method='ffill')  # 填充缺失值
            
            return df
            
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return pd.DataFrame()

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        # 计算移动平均线
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        
        # 计算DIF和DEA
        df['dif'] = df['ema_fast'] - df['ema_slow']
        df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
        
        # 计算MACD柱状图
        df['macd_hist'] = 2 * (df['dif'] - df['dea'])
        
        return df

    def generate_trade_signals(self, df):
        """生成买卖信号"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # 金叉买入信号：DIF 从下向上穿过 DEA
        for i in range(1, len(df)):
            if df['dif'].iloc[i] > df['dea'].iloc[i] and df['dif'].iloc[i-1] <= df['dea'].iloc[i-1]:
                signals['signal'].iloc[i] = 1
            # 死叉卖出信号：DIF 从上向下穿过 DEA
            elif df['dif'].iloc[i] < df['dea'].iloc[i] and df['dif'].iloc[i-1] >= df['dea'].iloc[i-1]:
                signals['signal'].iloc[i] = -1
        
        return signals

    def calculate_performance(self, df, signals):
        """计算交易策略的表现和收益率"""
        initial_capital = 10000.0
        cash = initial_capital
        shares = 0
        in_position = False
        trades = []
        
        for date in df.index:
            # 获取当前价格和信号
            price = df.loc[date, 'close']
            signal = signals.loc[date, 'signal']
            
            # 买入信号
            if signal == 1 and not in_position:
                shares = cash / price
                cash = 0
                in_position = True
                trades.append({
                    'date': date,
                    'action': '买入',
                    'price': price,
                    'shares': shares
                })
            # 卖出信号
            elif signal == -1 and in_position:
                cash = shares * price
                trades.append({
                    'date': date,
                    'action': '卖出',
                    'price': price,
                    'cash': cash
                })
                shares = 0
                in_position = False
        
        # 如果最后还持有股票，卖出计算最终收益
        if in_position:
            final_price = df.iloc[-1]['close']
            cash = shares * final_price
            trades.append({
                'date': df.index[-1],
                'action': '最终卖出',
                'price': final_price,
                'cash': cash
            })
        
        # 计算收益率
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'trades': trades
        }

    def plot_macd_strategy(self, df, signals, save_path, ts_code):
        """绘制MACD策略图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 绘制价格和买卖信号
        ax1.plot(df.index, df['close'], label='收盘价', color='blue')
        
        # 标记买入点
        buy_signals = signals[signals['signal'] == 1]
        ax1.scatter(buy_signals.index, df.loc[buy_signals.index]['close'], 
                   marker='^', color='green', s=100, label='买入信号')
        
        # 标记卖出点
        sell_signals = signals[signals['signal'] == -1]
        ax1.scatter(sell_signals.index, df.loc[sell_signals.index]['close'], 
                   marker='v', color='red', s=100, label='卖出信号')
        
        ax1.set_title(f'{ts_code} 股票价格与MACD交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        
        # 绘制MACD指标
        ax2.plot(df.index, df['dif'], label='DIF', color='blue')
        ax2.plot(df.index, df['dea'], label='DEA', color='orange')
        ax2.bar(df.index, df['macd_hist'], label='MACD柱状图', color='gray')
        ax2.set_title('MACD指标')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('MACD值')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def format_result(self, ts_code, signals, performance):
        """格式化分析结果为Markdown"""
        # 基本信息
        result = f"### {ts_code} MACD交易策略分析（过去一年）\n\n"
        result += f"**初始资金**: ¥{performance['initial_capital']:.2f}\n"
        result += f"**最终资金**: ¥{performance['final_value']:.2f}\n"
        result += f"**总收益率**: {performance['total_return']:.2f}%\n\n"
        
        # 交易记录
        result += "## 交易记录\n"
        result += "| 日期       | 交易类型 | 价格    | 数量/金额 |\n"
        result += "|------------|----------|---------|-----------|\n"
        
        for trade in performance['trades']:
            date_str = trade['date'].strftime('%Y-%m-%d')
            action = trade['action']
            price = trade['price']
            
            if 'shares' in trade:
                shares = trade['shares']
                detail = f"{shares:.2f} 股"
            else:
                cash = trade['cash']
                detail = f"¥{cash:.2f}"
            
            result += f"| {date_str} | {action} | {price:.2f} | {detail} |\n"
        
        result += "\n## 交易信号统计\n"
        result += f"- 买入信号次数: {len(signals[signals['signal'] == 1])}\n"
        result += f"- 卖出信号次数: {len(signals[signals['signal'] == -1])}\n"
        
        return result

# ====== ARIMA 股票预测工具类 ======
@register_tool('arima_stock')
class ArimaStockTool(BaseTool):
    """
    使用ARIMA模型预测股票未来N天价格的工具
    """
    description = '使用ARIMA模型预测股票未来N天的价格，从MySQL获取历史数据进行建模'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，例如：600519.SH、000858.SZ等',
            'required': True
        },
        {
            'name': 'n',
            'type': 'integer',
            'description': '预测未来的天数，默认为7天',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import os, time
        args = json.loads(params)
        ts_code = args['ts_code']
        n = args.get('n', 7)
        
        try:
            # 从MySQL获取历史数据
            df = self.get_historical_data(ts_code)
            if df.empty:
                return f"未找到股票代码 {ts_code} 的历史数据"
            
            # 使用ARIMA建模并预测
            forecast, model_summary = self.arima_forecast(df, n)
            
            # 生成可视化图表
            save_dir = os.path.join(os.path.dirname(__file__), 'stock_charts')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'arima_forecast_{ts_code}_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            img_path = os.path.join('stock_charts', filename)
            
            # 生成预测图表
            self.plot_forecast(df, forecast, save_path, ts_code)
            
            # 准备返回结果
            forecast_md = self.format_forecast_result(forecast)
            result_md = f"### {ts_code} 未来 {n} 天价格预测\n\n"
            result_md += forecast_md
            result_md += f"\n![{ts_code} 价格预测]({img_path})"
            
            return result_md
            
        except Exception as e:
            error_msg = f"ARIMA预测失败: {str(e)}"
            print(error_msg)
            return error_msg

    def get_historical_data(self, ts_code):
        """从MySQL获取股票历史数据"""
        # 从环境变量获取MySQL连接信息
        mysql_config = get_mysql_config()
        database = 'stock_data'
        
        try:
            engine = create_engine(
                f'mysql+pymysql://{mysql_config["user"]}:{mysql_config["password"]}@{mysql_config["host"]}:{mysql_config["port"]}/{database}?charset=utf8mb4',
                connect_args={'connect_timeout': 10}
            )
            
            # 计算日期范围：截止到今天的前一年
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 查询股票数据
            sql = text("""
                SELECT trade_date, close FROM all_symbols 
                WHERE stock_code = :code 
                AND trade_date BETWEEN :start AND :end
                ORDER BY trade_date
            """)
            
            df = pd.read_sql(sql, engine, params={'code': ts_code, 'start': start_date, 'end': end_date})
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df = df.asfreq('B')  # 使用工作日频率
            df = df.fillna(method='ffill')  # 填充缺失值
            
            return df
            
        except Exception as e:
            print(f"获取历史数据失败: {str(e)}")
            return pd.DataFrame()

    def arima_forecast(self, df, n):
        """使用ARIMA(5,1,5)模型进行预测"""
        try:
            # 构建ARIMA模型
            model = ARIMA(df['close'], order=(5, 1, 5))
            model_fit = model.fit()
            
            # 预测未来n天
            forecast = model_fit.forecast(steps=n)
            
            return forecast, model_fit.summary()
            
        except Exception as e:
            print(f"ARIMA建模失败: {str(e)}")
            raise

    def plot_forecast(self, df, forecast, save_path, ts_code):
        """绘制历史数据和预测结果图表"""
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        plt.plot(df.index, df['close'], label='历史收盘价', color='blue')
        
        # 生成预测日期
        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(forecast), freq='B')
        
        # 绘制预测数据
        plt.plot(forecast_dates, forecast, label='预测价格', color='red', linestyle='--', marker='o')
        
        plt.title(f'{ts_code} 股票价格历史数据与未来预测')
        plt.xlabel('日期')
        plt.ylabel('收盘价')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def format_forecast_result(self, forecast):
        """格式化预测结果为Markdown表格"""
        # 生成预测日期
        last_date = datetime.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
        
        # 过滤工作日
        business_dates = []
        i = 0
        while len(business_dates) < len(forecast):
            current_date = last_date + timedelta(days=i+1)
            if current_date.weekday() < 5:  # 0-4 为工作日
                business_dates.append(current_date)
            i += 1
        
        # 构建Markdown表格
        result = "| 日期       | 预测价格 |\n|------------|----------|\n"
        for date, price in zip(business_dates, forecast):
            result += f"| {date.strftime('%Y-%m-%d')} | {price:.2f} |\n"
        
        return result

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    description = '对于生成的SQL，进行SQL查询，并自动可视化股票数据'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        from sqlalchemy import text
        args = json.loads(params)
        sql_input = args['sql_input']
        print('sql_input=', sql_input)
        database = args.get('database', 'stock_data')
        
        mysql_config = get_mysql_config()
        
        try:
            engine = create_engine(
                f'mysql+pymysql://{mysql_config["user"]}:{mysql_config["password"]}@{mysql_config["host"]}:{mysql_config["port"]}/{database}?charset=utf8mb4',
                connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
            )
            df = pd.read_sql(text(sql_input), engine)
            print('DataFrame result:', df)
            md = df.head(10).to_markdown(index=False)
            save_dir = os.path.join(os.path.dirname(__file__), 'stock_charts')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'stock_chart_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            generate_stock_chart(df, save_path)
            img_path = os.path.join('stock_charts', filename)
            img_md = f'![股票图表]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            error_msg = f"数据库连接或查询失败: {str(e)}"
            error_msg += "\n\n请检查："
            error_msg += "\n1. MySQL 服务是否已启动"
            error_msg += f"\n2. 当前使用的连接参数: 用户名={mysql_config['user']}, 主机={mysql_config['host']}, 端口={mysql_config['port']}, 数据库={database}"
            error_msg += "\n3. 您可以通过设置环境变量来配置连接: MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT"
            error_msg += "\n4. stock_data 数据库是否已创建"
            error_msg += "\n5. 表 all_symbols 是否存在且有数据"
            print(error_msg)
            return error_msg

# ========== 股票数据可视化函数 ========== 
def generate_stock_chart(df_sql, save_path):
    columns = df_sql.columns
    
    if 'trade_date' in df_sql.columns and 'close' in df_sql.columns:
        plt.figure(figsize=(12, 6))
        
        if 'stock_code' in df_sql.columns:
            grouped = df_sql.groupby('stock_code')
            for name, group in grouped:
                group['trade_date'] = pd.to_datetime(group['trade_date'])
                group = group.sort_values('trade_date')
                plt.plot(group['trade_date'], group['close'], label=name)
        else:
            df_sql['trade_date'] = pd.to_datetime(df_sql['trade_date'])
            df_sql = df_sql.sort_values('trade_date')
            plt.plot(df_sql['trade_date'], df_sql['close'], label='收盘价')
        
        plt.title('股票价格趋势')
        plt.xlabel('交易日期')
        plt.ylabel('价格')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        x = np.arange(len(df_sql))
        num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()
        object_columns = df_sql.select_dtypes(include='O').columns.tolist()
        
        plt.figure(figsize=(10, 6))
        
        if len(object_columns) > 0:
            for column in num_columns:
                plt.bar(df_sql[object_columns[0]], df_sql[column], label=column)
        else:
            for column in columns:
                plt.plot(x, df_sql[column], label=column)
        
        plt.title('股票数据统计')
        plt.xlabel('数据点')
        plt.ylabel('数值')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ====== 初始化股票助手服务 ======
def init_agent_service():
    llm_cfg = {
        'model': 'qwen-turbo',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票查询分析助手',
            description='股票历史数据查询、分析与可视化，以及使用ARIMA模型进行价格预测和MACD交易策略分析',
            system_message=system_prompt,
            function_list=['exc_sql', 'get_stock_news', 'arima_stock', 'macd_stock', 'rag_stock_knowledge'],
        )
        print("股票助手初始化成功！")
        return bot
    except Exception as e:
        print(f"股票助手初始化失败: {str(e)}")
        raise

def app_tui():
    try:
        bot = init_agent_service()
        messages = []
        while True:
            try:
                query = input('请输入您的股票查询问题: ')
                file = input('文件路径（若无文件请按回车）: ').strip()
                
                if not query:
                    print('查询问题不能为空！')
                    continue
                    
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                response = []
                for res in bot.run(messages):
                    print('助手响应:', res)
                    response.extend(res)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    try:
        print("正在启动股票查询分析助手 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '查询贵州茅台(600519.SH)最近一个月的收盘价趋势',
                '比较四只股票的成交量情况',
                '计算五粮液(000858.SZ)的最大涨跌幅',
                '分析最近一周所有股票的平均涨幅',
                '查询所有股票的最高价和最低价统计',
                '获取贵州茅台最新相关新闻',
                '查看中芯国际最近的新闻动态',
                '预测贵州茅台未来7天的价格',
                '使用ARIMA模型预测五粮液未来10天价格',
                '分析贵州茅台过去一年的MACD交易策略',
                '计算五粮液过去一年的MACD买卖点和收益率'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    app_gui()
