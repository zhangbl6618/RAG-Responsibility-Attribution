import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, AsyncAzureOpenAI

class OpenAIGeneratorError(Exception):
    """Custom exception for OpenAI Generator errors"""
    pass

class OpenAIGenerator:
    """Async OpenAI text generator with batch processing support"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI generator
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self._validate_config(config)
        self.model_name = config["generator_model"]
        self.list = config.get("generate_list", False)
        self.list_size = config.get("list_size", 5) if self.list else None
        self.batch_size = config.get("generator_batch_size", 10)
        self.generation_params = config.get("generation_params", {})
        
        self.client = self._initialize_client(config["openai_setting"])
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration parameters"""
        if not config.get("generator_model"):
            raise OpenAIGeneratorError("generator_model is required but not provided.")
        
        if "openai_setting" not in config:
            raise OpenAIGeneratorError("openai_setting is required but not provided.")
    
    def _initialize_client(self, openai_setting: Dict[str, Any]) -> AsyncOpenAI:
        """Initialize the OpenAI client based on settings"""
        setting = openai_setting.copy()
        
        if not setting.get("api_key"):
            setting["api_key"] = os.getenv("OPENAI_API_KEY")
            if not setting["api_key"]:
                raise OpenAIGeneratorError("OpenAI API key not found in config or environment variables")
        
        if setting.pop("api_type", None) == "azure":
            return AsyncAzureOpenAI(**setting)
        else:
            return AsyncOpenAI(**setting)
    
    async def get_response(self, messages: List[Dict[str, str]], **params) -> str:
        """
        Get a single response from OpenAI API
        
        Args:
            messages: List of message dictionaries
            **params: Additional parameters for the API call
            
        Returns:
            Generated text content
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages, 
                **params
            )
            if not response or not response.choices:
                raise OpenAIGeneratorError("Invalid response or no choices in the response")
            
            content = response.choices[0].message.content
            if content is None:
                raise OpenAIGeneratorError("No content in response")
                
            return content
            
        except Exception as e:
            raise OpenAIGeneratorError(f"Failed to get response: {str(e)}")
    
    async def get_batch_response(self, input_list: List[List[Dict[str, str]]], 
                               batch_size: int, **params) -> List[str]:
        """
        Get batch responses from OpenAI API with proper batching
        
        Args:
            input_list: List of message lists
            batch_size: Size of each batch
            **params: Additional parameters for the API calls
            
        Returns:
            List of generated responses
        """
        if not input_list:
            return []
        
        all_results = []
        
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i + batch_size]
            
            tasks = [self.get_response(messages, **params) for messages in batch]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        all_results.append("") 
                    else:
                        all_results.append(result)
                        
            except Exception as e:
                all_results.extend([""] * len(batch))
        
        return all_results
    
    def generate(self, input_list: List[List[Dict[str, str]]], 
                batch_size: Optional[int] = None, **params) -> List[str]:
        """
        Generate responses for a list of inputs
        
        Args:
            input_list: List of message lists to process
            batch_size: Optional batch size override
            **params: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if not input_list:
            return []
        
        batch_size = batch_size or self.batch_size
        
        generation_params = self.generation_params.copy()
        generation_params.update(params)
        
        if "n" not in generation_params:
            generation_params["n"] = 1
        
        try:
            return asyncio.run(
                self.get_batch_response(input_list, batch_size, **generation_params)
            )
        except Exception as e:
            raise OpenAIGeneratorError(f"Generation failed: {str(e)}")