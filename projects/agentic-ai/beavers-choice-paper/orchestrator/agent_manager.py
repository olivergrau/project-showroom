# agent_manager.py

from typing import Protocol

class AgentInterface(Protocol):
    """Protocol defining the interface all agents must implement."""
    def run(self, *args, **kwargs):
        ...

class AgentManager:
    """Manages agent instances and their verbosity settings."""
    
    def __init__(self, parser_agent: AgentInterface, inventory_agent: AgentInterface,
                 quote_agent: AgentInterface, order_agent: AgentInterface,
                 reporting_agent: AgentInterface, verbosity: int = 1):
        self.parser_agent = parser_agent
        self.inventory_agent = inventory_agent
        self.quote_agent = quote_agent
        self.order_agent = order_agent
        self.reporting_agent = reporting_agent
        self.verbosity = verbosity
        
        if verbosity >= 2:
            self._set_agent_verbosity(2)
    
    def _set_agent_verbosity(self, level: int):
        """Set verbosity level for all agents that support it."""
        agents = [self.parser_agent, self.inventory_agent, self.quote_agent, 
                 self.order_agent, self.reporting_agent]
        
        for agent in agents:
            if hasattr(agent, 'agent') and hasattr(agent.agent, 'verbosity_level'):
                agent.agent.verbosity_level = level
