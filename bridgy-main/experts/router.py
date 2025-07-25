import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from .ai_pods_expert import AIPodExpert
from .general_expert import GeneralExpert


import logging
from config import setup_langsmith

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Initialize LangSmith
setup_langsmith()

class ExpertRouter:
    def __init__(self):
        logger.error("initialize Expert Router")
        # Using OpenAI-compatible API for vLLM or remote LLM service
        self.llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "gemma-2-9b"),
            base_url=os.getenv("LLM_SERVICE_URL", "http://vllm-server:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", "llm-api-key"),
            temperature=0.0
        )

        # Initialize experts
        self.experts = {

            "ai_pods": AIPodExpert(),
            "general": GeneralExpert(),


        }

        # Router prompt template
        self.router_prompt = ChatPromptTemplate.from_template("""
        You are a router for a Cisco infrastructure assistant. Your job is to determine which expert should handle the user's question.

        Available experts:

        1. AI Pods Expert: For questions about:
           - Cisco AI Pods
           - AI compute infrastructure
           - ANY mention of LLM parameter sizes like 7B, 13B, 40B, 70B, etc.
           - ANY questions about model sizes, parameters, or AI hardware requirements
        2. General Expert: For any questions not related to AI Pods.

        Question: {question}

        Respond with just the expert name: "ai_pods" or "general"
        """)


        # Create router chain using the new RunnableSequence pattern
        self.router_chain = self.router_prompt | self.llm

    def route_and_respond(self, query: str) -> tuple[str, str]:
        try:
            logger.info("Using chain of thought to determine expert")
            expert_choice = self._determine_expert_with_cot(query)
            logger.info(f"Chain of thought selected: {expert_choice}")

            if expert_choice == "ai_pods":
                try:
                    logger.info("Routing to AI Pods Expert")
                    return self.experts["ai_pods"].get_response(query), "AI Pods Expert"
                except Exception as e:
                    logger.error(f"AI Pods Expert error: {str(e)}")
                    return f"AI Pods Expert Error: {str(e)}", "System"




            else:
                try:
                    logger.info("Routing to General Expert")
                    return self.experts["general"].get_response(query), "General Expert"
                except Exception as e:
                    logger.error(f"General Expert error: {str(e)}")
                    return f"General Expert Error: {str(e)}", "System"

        except Exception as routing_error:
            logger.error(f"Error in routing logic: {str(routing_error)}")
            return self._basic_routing_fallback(query)

    def _determine_expert_with_cot(self, query: str) -> str:
        try:
            response = self.router_chain.invoke({"question": query})
            
            # Extract content from response
            if hasattr(response, 'content'):
                expert_choice = response.content.strip().lower()
            elif isinstance(response, dict) and 'content' in response:
                expert_choice = response['content'].strip().lower()
            elif isinstance(response, str):
                expert_choice = response.strip().lower()
            else:
                expert_choice = str(response).strip().lower()
                
            logger.debug(f"Router chain response: {expert_choice}")

            if expert_choice in ["ai_pods", "general"]:
                return expert_choice

            last_words = expert_choice.split()[-5:]
            if "ai_pods" in last_words or "ai pods" in ' '.join(last_words):
                return "ai_pods"


            elif "general" in last_words:
                return "general"


                # Former Nexus Dashboard queries now route to General Expert

            return "general"

        except Exception as e:
            logger.error(f"Error in chain of thought: {str(e)}")
            # Route based on content
            query_lower = query.lower()
            
            # Check for AI Pods queries
            if "ai pods" in query_lower or "ai pod" in query_lower or any(word in query_lower for word in ["llm", "model", "parameter", "parameters"]) or any(size in query_lower for size in ["7b", "13b", "40b", "70b", "80b"]):
                return "ai_pods"
            # All other queries go to General Expert
            else:
                return "general"

    def _basic_routing_fallback(self, query: str) -> tuple[str, str]:
        # Former Intersight queries now go to General Expert
        if "intersight" in query.lower() or any(word in query.lower() for word in ["server", "servers", "hardware", "datacenter"]):
            try:
                return self.experts["general"].get_response(query), "General Expert"
            except Exception as e:
                logger.error(f"General Expert error: {str(e)}")
                return f"I'm sorry, I encountered multiple errors trying to process your request.", "System"
        elif self._is_ai_pods_query(query):
            try:
                return self.experts["ai_pods"].get_response(query), "AI Pods Expert"
            except:
                return "I'm sorry, I encountered an error with the AI Pods expert.", "System"
        elif "nexus" in query.lower() or "dashboard" in query.lower() or "fabric" in query.lower() or "switch" in query.lower():
            try:
                return self.experts["general"].get_response(query), "General Expert"
            except Exception as e:
                logger.error(f"General Expert error: {str(e)}")
                return "I'm sorry, I encountered an error processing your request.", "System"
        elif self._is_infrastructure_query(query):
            try:
                return self.experts["general"].get_response(query), "General Expert"
            except Exception as e:
                logger.error(f"General Expert error: {str(e)}")
                return "I'm sorry, I encountered an error processing your request.", "System"
        else:
            try:
                return self.experts["general"].get_response(query), "General Expert"
            except:
                return "I'm sorry, I encountered an error processing your request.", "System"



    def _is_ai_pods_query(self, query: str) -> bool:
        query_lower = query.lower()

        llm_size_patterns = [
            "7b", "7 b", "13b", "13 b", "40b", "40 b", "70b", "70 b",
            "billion parameter", "billion parameters", "b model", "b llm"
        ]
        if any(size_pattern in query_lower for size_pattern in llm_size_patterns):
            return True

        general_knowledge_indicators = [
            "why is", "what is", "how does", "explain", 
            "tell me about", "describe", "definition of"
        ]

        if any(indicator in query_lower for indicator in general_knowledge_indicators):
            if any(cisco_term in query_lower for cisco_term in ["cisco", "ai pod", "ai pods", "llm", "large language model"]):
                return True
            else:
                ai_hardware_terms = ["hardware for llm", "gpu for llm", "infrastructure for llm", "server for ai"]
                if any(term in query_lower for term in ai_hardware_terms):
                    return True
                return False

        ai_pods_keywords = [
            "ai pods", "ai pod", "aipod", "aipods", 
            "llm", "large language model", "ml model",
            "machine learning", "deep learning", "neural network",
            "gpu cluster", "ai infrastructure", "ml infrastructure",
            "ai compute", "ml compute", "inference", 
            "gpu for", "hardware for", "server for ai", "hardware recommendation",
            "what should i buy", "which pod", "recommendation for llm"
        ]
        return any(keyword in query_lower for keyword in ai_pods_keywords)

    def _is_server_inventory_query(self, query: str) -> bool:
        query = query.lower().strip()

        environment_patterns = [
            query.startswith("what") and ("environment" in query or "servers" in query),
            "running in " in query and ("environment" in query or "my" in query),
            "servers do i have" in query,
            "in my environment" in query,
            "show" in query and ("servers" in query or "infrastructure" in query),
            "list" in query and ("servers" in query or "infrastructure" in query),
            "what's" in query and "running" in query,
            "what is" in query and "running" in query,
            "deployed" in query and "servers" in query
        ]

        return any(environment_patterns)



    def _is_infrastructure_query(self, query: str) -> bool:
        """Check if a query is related to infrastructure across multiple systems."""
        query_lower = query.lower()
        
        # Check for explicit mentions of both systems
        if ("intersight" in query_lower and "nexus" in query_lower) or \
           ("intersight" in query_lower and "dashboard" in query_lower):
            return True
            
        # Check for switch-related queries that might span multiple systems
        switch_patterns = [
            "switches" in query_lower and "environment" in query_lower,
            "switch" in query_lower and "environment" in query_lower,
            "switches" in query_lower and "running" in query_lower,
            "switch" in query_lower and "running" in query_lower,
            "what switches" in query_lower,
            "network device" in query_lower and "environment" in query_lower
        ]
        
        # Exclude fabric-related queries as they should go to Nexus Dashboard Expert
        if ("fabric" in query_lower or "vlan" in query_lower) and not "environment" in query_lower:
            return False
            
        # Exclude server-related queries as they should go to Intersight Expert
        if "server" in query_lower or "servers" in query_lower or "firmware" in query_lower or "gpu" in query_lower:
            return False
            
        if any(switch_patterns):
            return True
            
        return False

    def get_response(self, question: str) -> str:
        try:
            # Determine which expert should handle this question
            expert_name = self._route_question(question)
            logger.info(f"Chain of thought selected: {expert_name}")
            
            # Route to the appropriate expert
            logger.info(f"Routing to {expert_name.replace('_', ' ').title()} Expert")
            expert = self.experts[expert_name]
            response = expert.get_response(question)
            
            # Extract just the content from the response if needed
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            elif isinstance(response, str):
                return response
            else:
                # Try to convert the response to a string if it's not already
                return str(response)
        except Exception as e:
            logger.error(f"{expert_name.replace('_', ' ').title()} Expert error: {str(e)}")
            
            # Fall back to general expert if specific expert fails
            logger.info("Falling back to General Expert due to error")
            try:
                return self.experts["general"].get_response(question)
            except Exception as fallback_error:
                logger.error(f"Fallback to General Expert also failed: {str(fallback_error)}")
                return f"I'm sorry, but I encountered an error while processing your question: {str(e)}"
