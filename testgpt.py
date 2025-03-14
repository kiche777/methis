import sys
import asyncio
import io
import contextlib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QScrollArea, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal

# Import your agent and LLM
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Initialize the LLM (global)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Worker thread to run the asynchronous agent code
class Worker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        
    def run(self):
        result = self.execute_agent(self.prompt)
        self.finished.emit(result)
        
    def execute_agent(self, prompt):
        f = io.StringIO()
        # Redirect stdout so any print output is captured.
        with contextlib.redirect_stdout(f):
            asyncio.run(self.run_agent(prompt))
        return f.getvalue()
    
    async def run_agent(self, prompt):
        agent = Agent(task=prompt, llm=llm)
        await agent.run(max_steps=30)

# Main window UI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern UI App")
        self.resize(800, 600)
        self.initUI()
        
    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        mainLayout = QHBoxLayout()
        centralWidget.setLayout(mainLayout)
        
        # LEFT: Output area and prompt input
        leftLayout = QVBoxLayout()
        mainLayout.addLayout(leftLayout, 3)
        
        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)
        leftLayout.addWidget(self.outputText)
        
        promptLayout = QHBoxLayout()
        self.inputLine = QLineEdit()
        self.sendButton = QPushButton("Send")
        self.sendButton.clicked.connect(self.handleSend)
        promptLayout.addWidget(self.inputLine)
        promptLayout.addWidget(self.sendButton)
        leftLayout.addLayout(promptLayout)
        
        # RIGHT: History panel
        rightLayout = QVBoxLayout()
        mainLayout.addLayout(rightLayout, 1)
        
        self.historyWidget = QWidget()
        self.historyLayout = QVBoxLayout()
        self.historyWidget.setLayout(self.historyLayout)
        
        self.historyScroll = QScrollArea()
        self.historyScroll.setWidgetResizable(True)
        self.historyScroll.setWidget(self.historyWidget)
        rightLayout.addWidget(self.historyScroll)
        
    def handleSend(self):
        prompt = self.inputLine.text().strip()
        if not prompt:
            return
        
        # Create a history entry with a check box and a read-only text field.
        historyEntry = QWidget()
        entryLayout = QHBoxLayout()
        historyEntry.setLayout(entryLayout)
        checkbox = QCheckBox()
        promptDisplay = QLineEdit()
        promptDisplay.setText(prompt)
        promptDisplay.setReadOnly(True)
        entryLayout.addWidget(checkbox)
        entryLayout.addWidget(promptDisplay)
        self.historyLayout.addWidget(historyEntry)
        
        self.inputLine.clear()
        
        # Launch the agent code in a worker thread.
        self.worker = Worker(prompt)
        self.worker.finished.connect(self.displayResult)
        self.worker.start()
        
    def displayResult(self, result):
        self.outputText.append(result)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())