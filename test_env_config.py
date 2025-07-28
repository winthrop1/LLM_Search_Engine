"""
Test script to verify environment configuration is working properly.
"""

import os
from dotenv import load_dotenv
from src.llm_router import LLMRouter

def test_env_configuration():
    """Test that all environment variables are being read correctly."""
    print("🧪 Testing Environment Configuration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Test reading individual environment variables
    print("📋 Environment Variables:")
    env_vars = [
        "LLM_PROVIDER",
        "LLM_MODEL", 
        "MAX_TOKENS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        # Mask API keys for security
        if "API_KEY" in var and value != "Not set":
            display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        else:
            display_value = value
        print(f"   {var}: {display_value}")
    
    print(f"\n🤖 Testing LLM Router Configuration:")
    
    try:
        # Test LLM Router initialization with environment variables
        llm_router = LLMRouter()  # Should read all values from environment
        
        provider_info = llm_router.get_provider_info()
        print(f"✅ LLM Router initialized successfully:")
        for key, value in provider_info.items():
            if key == "available_providers":
                print(f"   {key}: {', '.join(value)}")
            else:
                print(f"   {key}: {value}")
        
        # Test that max_tokens is correctly configured
        print(f"\n🔧 Configuration Test:")
        print(f"   Provider from env: {os.getenv('LLM_PROVIDER')}")
        print(f"   Model from env: {os.getenv('LLM_MODEL')}")
        print(f"   Max tokens from env: {os.getenv('MAX_TOKENS')}")
        print(f"   Router max_tokens: {llm_router.max_tokens}")
        
        # Verify max_tokens is correctly set
        assert llm_router.max_tokens == int(os.getenv("MAX_TOKENS", "1000")), "max_tokens not correctly configured"
        print(f"✅ max_tokens correctly configured from environment")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing LLM Router: {e}")
        print(f"💡 This is expected if API keys are not valid")
        return False

def test_override_configuration():
    """Test that parameters can still be overridden programmatically."""
    print(f"\n🔄 Testing Configuration Override:")
    
    try:
        # Test overriding configuration
        custom_router = LLMRouter(
            provider="openai", 
            model="gpt-4", 
            max_tokens=500,
            api_key="test-key"
        )
        
        print(f"✅ Custom configuration works:")
        print(f"   Provider: {custom_router.provider_name}")
        print(f"   Model: {custom_router.model}")
        print(f"   Max tokens: {custom_router.max_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with custom configuration: {e}")
        return False

if __name__ == "__main__":
    env_test = test_env_configuration()
    override_test = test_override_configuration()
    
    print(f"\n🎯 Test Results:")
    print(f"   Environment configuration: {'✅ PASS' if env_test else '❌ FAIL'}")
    print(f"   Override configuration: {'✅ PASS' if override_test else '❌ FAIL'}")
    
    if env_test and override_test:
        print(f"\n🎉 All configuration tests passed!")
    else:
        print(f"\n⚠️  Some tests failed - check your .env file configuration")