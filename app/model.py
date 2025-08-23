# model.py - OpenAI Version

import requests
import os
import time
import base64
from PIL import Image
import io
import json
from openai import OpenAI

# Load API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Put this in a .env file or environment settings

# Initialize OpenAI client
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

def get_openai_client():
    """Get OpenAI client"""
    global client
    if not client:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def encode_image_to_base64(image_path):
    """Convert image to base64 for API submission"""
    try:
        with open(image_path, "rb") as image_file:
            # Resize image if too large to reduce API payload
            img = Image.open(image_file)
            
            # Resize if image is too large (max 1024x1024 for better performance)
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def analyze_image_with_vision_model(image_path, image_type="crop_problem"):
    """Use OpenAI's GPT-4 Vision to describe the image"""
    try:
        client = get_openai_client()
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Set prompt based on image type
        if image_type == "pesticide":
            prompt_text = "Analyze this pesticide image. Describe what you see including any text, labels, product names, and visual characteristics that might help identify the pesticide type."
        else:
            prompt_text = "Analyze this image. Describe what you see including any visible problems, diseases, or pests affecting the crop."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the free tier model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Vision model error: {str(e)}")
        return f"Error analyzing image: {str(e)}"

def analyze_text_with_llm(prompt, system_role="crop_problem"):
    """Use OpenAI's GPT model to analyze the description and provide recommendations"""
    try:
        client = get_openai_client()
        
        # Set system message based on the type of analysis
        if system_role == "pesticide":
            system_content = "You are an expert agricultural safety consultant specializing in pesticide identification and safety recommendations. Provide clear, accurate, and safety-focused advice."
        else:
            system_content = "You are an expert agricultural consultant specializing in crop problem identification and solutions. Provide clear, accurate, and practical advice for farmers."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the free tier model
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Text model error: {str(e)}")
        return f"Error generating analysis: {str(e)}"

def colorize_text(text):
    """Format text with colors (for display purposes)"""
    for word, color in zip(["Question", "Answer"], ["red", "green"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

def run_image_query(image_path, question_text, max_new_tokens=256, system_role="crop_problem"):
    """
    Complete image analysis pipeline using OpenAI:
    1. Use GPT-4 Vision to describe the image
    2. Use GPT-4 to analyze and provide recommendations
    """
    start_time = time.time()
    
    try:
        # Step 1: Get image description using OpenAI Vision
        print("Step 1: Analyzing image with OpenAI Vision...")
        image_description = analyze_image_with_vision_model(image_path, "crop_problem" if system_role != "pesticide" else "pesticide")
        
        # Step 2: Create comprehensive prompt for text analysis
        analysis_prompt = f"""
        Image Description: {image_description}
        
        Question: {question_text}
        
        Based on the image description above, please provide a structured analysis with:
        
        1. **Pesticide Identification**: What type of pesticide is this?
        2. **Safety Classification**: Rate as Safe/Caution/High Risk
        3. **Active Ingredients**: List identifiable active ingredients
        4. **Usage Recommendations**: How should this be applied safely?
        5. **Safety Precautions**: What protective measures are needed?
        6. **Confidence Level**: How confident are you in this identification (percentage)?
        
        Format your response clearly with each section labeled.
        """
        
        # Step 3: Get detailed analysis from OpenAI
        print("Step 2: Generating analysis with OpenAI...")
        analysis_result = analyze_text_with_llm(analysis_prompt, system_role)
        
        elapsed_time = round(time.time() - start_time, 2)
        
        return analysis_result, elapsed_time
        
    except Exception as e:
        elapsed_time = round(time.time() - start_time, 2)
        error_msg = f"Analysis failed: {str(e)}"
        print(f"Complete analysis error: {error_msg}")
        return error_msg, elapsed_time

def extract_pesticide_info_from_filename(filename):
    """Extract pesticide information based on filename patterns"""
    filename_lower = filename.lower()
    
    # Common pesticide name patterns
    pesticide_patterns = {
        'neem': {'type': 'Neem Oil', 'safety': 'Safe', 'confidence': 85},
        'roundup': {'type': 'Glyphosate (Roundup)', 'safety': 'Caution', 'confidence': 90},
        'malathion': {'type': 'Malathion', 'safety': 'High Risk', 'confidence': 88},
        'chlorpyrifos': {'type': 'Chlorpyrifos', 'safety': 'High Risk', 'confidence': 87},
        'bt': {'type': 'Bt (Bacillus thuringiensis)', 'safety': 'Safe', 'confidence': 80},
        'copper': {'type': 'Copper-based Fungicide', 'safety': 'Caution', 'confidence': 75},
        'sulfur': {'type': 'Sulfur-based Fungicide', 'safety': 'Safe', 'confidence': 78}
    }
    
    for pattern, info in pesticide_patterns.items():
        if pattern in filename_lower:
            return info
    
    return {'type': 'Unknown Pesticide', 'safety': 'Caution', 'confidence': 50}

# Alternative simple analysis function for testing
def simple_pesticide_analysis(image_path, filename):
    """Simple fallback analysis without external APIs"""
    
    # Get basic info from filename
    file_info = extract_pesticide_info_from_filename(filename)
    
    # Create mock detailed analysis
    analysis = {
        'pesticide': file_info['type'],
        'safety': file_info['safety'],
        'confidence': file_info['confidence'],
        'recommendation': generate_recommendation(file_info['type'], file_info['safety']),
        'active_ingredients': get_common_ingredients(file_info['type']),
        'filename': filename
    }
    
    return analysis

def generate_recommendation(pesticide_type, safety_level):
    """Generate safety recommendations based on pesticide type and safety level"""
    
    base_recommendations = {
        'Safe': "This is considered a safer option for agricultural use. Apply according to label instructions. Suitable for integrated pest management programs.",
        'Caution': "Use with moderate caution. Wear protective equipment including gloves and eye protection. Avoid application during windy conditions.",
        'High Risk': "This is a high-risk pesticide. Use full protective equipment including respirator, gloves, and protective clothing. Apply only when absolutely necessary and follow all safety protocols."
    }
    
    specific_recommendations = {
        'Neem Oil': "Apply during cooler parts of the day. Can be used up to harvest. Mix with water as directed on label.",
        'Glyphosate': "Non-selective herbicide - will kill all vegetation. Use carefully around desired plants. Do not spray on windy days.",
        'Bt': "Target-specific biological pesticide. Safe for beneficial insects when used properly. Most effective on young larvae."
    }
    
    recommendation = base_recommendations.get(safety_level, "Follow all label instructions carefully.")
    
    # Add specific recommendations if available
    for pest_type, specific_rec in specific_recommendations.items():
        if pest_type.lower() in pesticide_type.lower():
            recommendation += f" {specific_rec}"
            break
    
    return recommendation

def get_common_ingredients(pesticide_type):
    """Get common active ingredients for pesticide types"""
    
    ingredient_map = {
        'Neem Oil': ['Azadirachtin', 'Neem oil extract'],
        'Glyphosate': ['Glyphosate', 'Isopropylamine salt'],
        'Malathion': ['Malathion', 'Organophosphate compound'],
        'Chlorpyrifos': ['Chlorpyrifos', 'Organophosphate insecticide'],
        'Bt': ['Bacillus thuringiensis', 'Bacterial proteins'],
        'Copper': ['Copper sulfate', 'Copper hydroxide'],
        'Sulfur': ['Elemental sulfur', 'Sulfur compounds']
    }
    
    for pest_type, ingredients in ingredient_map.items():
        if pest_type.lower() in pesticide_type.lower():
            return ingredients
    
    return ['Unknown active ingredients']

# Test function to verify API connectivity
def test_api_connection():
    """Test if OpenAI API is accessible"""
    try:
        if not OPENAI_API_KEY:
            return False, "API key not configured"
        
        client = get_openai_client()
        
        # Simple test request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Test connection - respond with 'OK'"}
            ],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            return True, "API connection successful"
        else:
            return False, "API returned empty response"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# Main analysis function that combines everything
def comprehensive_pesticide_analysis(image_path, filename):
    """
    Comprehensive pesticide analysis using OpenAI:
    1. Try AI-based analysis first
    2. Fall back to pattern-based analysis if AI fails
    3. Always provide safety recommendations
    """
    
    try:
        # Test API connection first
        api_available, api_message = test_api_connection()
        
        if api_available:
            # Use OpenAI-based analysis
            question = """Analyze this pesticide image and provide:
            1. Pesticide name and type
            2. Safety classification (Safe/Caution/High Risk)
            3. Active ingredients if visible
            4. Application recommendations
            5. Safety precautions needed"""
            
            ai_result, elapsed_time = run_image_query(image_path, question, max_new_tokens=400, system_role="pesticide")
            
            # Parse AI result into structured format
            return parse_comprehensive_result(ai_result, filename, elapsed_time, source="OpenAI")
            
        else:
            print(f"API not available: {api_message}. Using fallback analysis.")
            return simple_pesticide_analysis(image_path, filename)
            
    except Exception as e:
        print(f"Comprehensive analysis failed: {str(e)}")
        return simple_pesticide_analysis(image_path, filename)

def parse_comprehensive_result(ai_text, filename, analysis_time, source="OpenAI"):
    """Parse comprehensive AI result into structured data"""
    
    result = {
        'pesticide': 'Unknown Pesticide',
        'confidence': 75,
        'safety': 'Caution',
        'recommendation': ai_text,
        'active_ingredients': [],
        'filename': filename,
        'analysis_time': analysis_time,
        'source': source
    }
    
    ai_lower = ai_text.lower()
    
    # Enhanced pesticide detection
    pesticide_keywords = {
        'neem': {'name': 'Neem Oil Pesticide', 'safety': 'Safe', 'confidence': 90},
        'glyphosate': {'name': 'Glyphosate Herbicide', 'safety': 'Caution', 'confidence': 88},
        'roundup': {'name': 'Roundup (Glyphosate)', 'safety': 'Caution', 'confidence': 92},
        'malathion': {'name': 'Malathion Insecticide', 'safety': 'High Risk', 'confidence': 85},
        'chlorpyrifos': {'name': 'Chlorpyrifos Insecticide', 'safety': 'High Risk', 'confidence': 87},
        'bt': {'name': 'Bt Biological Pesticide', 'safety': 'Safe', 'confidence': 83},
        'copper': {'name': 'Copper-based Fungicide', 'safety': 'Caution', 'confidence': 80},
        'sulfur': {'name': 'Sulfur Fungicide', 'safety': 'Safe', 'confidence': 78},
        'pyrethrin': {'name': 'Pyrethrin Insecticide', 'safety': 'Caution', 'confidence': 82},
        'imidacloprid': {'name': 'Imidacloprid (Neonicotinoid)', 'safety': 'High Risk', 'confidence': 86}
    }
    
    # Check for pesticide types
    for keyword, info in pesticide_keywords.items():
        if keyword in ai_lower:
            result['pesticide'] = info['name']
            result['safety'] = info['safety']
            result['confidence'] = info['confidence']
            break
    
    # Enhanced safety detection
    if any(word in ai_lower for word in ['organic', 'natural', 'safe', 'low toxicity', 'biological']):
        result['safety'] = 'Safe'
        result['confidence'] = min(result['confidence'] + 5, 95)
    elif any(word in ai_lower for word in ['toxic', 'dangerous', 'harmful', 'restricted', 'high risk']):
        result['safety'] = 'High Risk'
        result['confidence'] = min(result['confidence'] + 10, 95)
    elif any(word in ai_lower for word in ['caution', 'moderate', 'warning', 'careful']):
        result['safety'] = 'Caution'
    
    # Extract confidence percentage if mentioned in AI response
    import re
    confidence_match = re.search(r'confidence[:\s]*(\d+)%', ai_lower)
    if confidence_match:
        result['confidence'] = int(confidence_match.group(1))
    
    # Extract active ingredients if mentioned
    ingredients_patterns = [
        'azadirachtin', 'glyphosate', 'malathion', 'chlorpyrifos', 
        'imidacloprid', 'cypermethrin', 'permethrin', 'copper sulfate',
        'bacillus thuringiensis', 'pyrethrin', 'carbaryl', 'diazinon'
    ]
    
    found_ingredients = [ing for ing in ingredients_patterns if ing in ai_lower]
    if found_ingredients:
        result['active_ingredients'] = found_ingredients
    else:
        result['active_ingredients'] = get_common_ingredients(result['pesticide'])
    
    return result

# Enhanced error handling wrapper
def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with retries"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                error_type = type(e).__name__
                if "timeout" in str(e).lower() or "RateLimitError" in error_type:
                    print(f"{error_type} on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Error on attempt {attempt + 1}: {str(e)}")
                    time.sleep(retry_delay)
            else:
                raise e
    
    return None

# Usage example function
def analyze_pesticide_image(image_path, filename=None):
    """
    Main function to analyze a pesticide image
    
    Args:
        image_path (str): Path to the pesticide image
        filename (str): Optional filename for additional context
    
    Returns:
        dict: Analysis results including pesticide type, safety level, and recommendations
    """
    if not filename:
        filename = os.path.basename(image_path)
    
    return comprehensive_pesticide_analysis(image_path, filename)

# Crop problem analysis function using OpenAI
def analyze_crop_problem_with_ai(image_path, filename):
    """
    Analyze crop problems using OpenAI vision model and provide preventive measures
    
    Args:
        image_path (str): Path to the crop problem image
        filename (str): Filename for reference
    
    Returns:
        dict: Analysis results with problem identification and preventive measures
    """
    try:
        # First, get image description using OpenAI Vision
        image_description = analyze_image_with_vision_model(image_path, "crop_problem")
        
        # Create prompt for text analysis to identify crop problems and solutions
        analysis_prompt = f"""
        Image Description: {image_description}
        
        Based on the image description above, please provide a comprehensive analysis for crop problem detection:
        
        1. **Problem Identification**: What specific crop problem or disease is visible in the image?
        2. **Problem Description**: Provide a detailed description of the problem
        3. **Confidence Level**: How confident are you in this identification (percentage)?
        4. **Preventive Measures**:
           - Immediate actions to take
           - Long-term prevention strategies
        5. **Treatment Solutions**:
           - Recommended treatments or interventions
           - Products that can be used
        6. **Additional Recommendations**: Any other important information for the farmer
        
        Format your response clearly with each section labeled.
        """
        
        # Get detailed analysis from OpenAI
        analysis_result = analyze_text_with_llm(analysis_prompt, "crop_problem")
        
        # Parse the result into structured format
        parsed_result = parse_crop_problem_result(analysis_result, filename)
        
        return parsed_result
        
    except Exception as e:
        print(f"Crop problem analysis error: {str(e)}")
        # Return fallback result
        return {
            'problem_name': 'Unknown Crop Problem',
            'description': 'Unable to analyze the image automatically. Please consult with agricultural experts.',
            'confidence': 50,
            'preventive_measures': {
                'immediate': ['Inspect plants regularly', 'Remove affected parts', 'Maintain good hygiene'],
                'long_term': ['Crop rotation', 'Proper irrigation', 'Use disease-resistant varieties']
            },
            'solutions': ['Consult local agricultural extension services', 'Apply appropriate fungicides/insecticides if needed'],
            'additional_recommendations': 'Always follow label instructions when using any agricultural products.'
        }

def parse_crop_problem_result(ai_text, filename):
    """Parse AI result into structured crop problem data"""
    # This is a simplified parser - in a real implementation, you'd want more robust parsing
    result = {
        'problem_name': 'Unknown Crop Problem',
        'description': ai_text,
        'confidence': 75,
        'preventive_measures': {
            'immediate': ['Inspect plants regularly', 'Remove affected parts', 'Maintain good hygiene'],
            'long_term': ['Crop rotation', 'Proper irrigation', 'Use disease-resistant varieties']
        },
        'solutions': ['Consult local agricultural extension services', 'Apply appropriate fungicides/insecticides if needed'],
        'additional_recommendations': 'Always follow label instructions when using any agricultural products.',
        'filename': filename
    }
    
    # Extract key information from the AI response
    ai_lower = ai_text.lower()
    
    # Try to identify specific problems based on keywords
    if 'rust' in ai_lower or 'spot' in ai_lower:
        result['problem_name'] = 'Leaf Spot Disease'
        result['description'] = 'Leaf spot disease characterized by dark spots on leaves. Often caused by fungal pathogens.'
        result['confidence'] = 85
        result['preventive_measures']['immediate'] = ['Remove infected leaves', 'Improve air circulation', 'Avoid overhead watering']
        result['preventive_measures']['long_term'] = ['Use resistant varieties', 'Apply fungicides as needed', 'Practice crop rotation']
        result['solutions'] = ['Apply copper-based fungicides', 'Use systemic fungicides for severe cases', 'Remove and destroy infected plant material']
    elif 'aphid' in ai_lower or 'insect' in ai_lower:
        result['problem_name'] = 'Aphid Infestation'
        result['description'] = 'Aphid infestation causing yellowing and curling of leaves. These small insects suck plant juices.'
        result['confidence'] = 80
        result['preventive_measures']['immediate'] = ['Spray with water to dislodge aphids', 'Use insecticidal soap', 'Introduce natural predators']
        result['preventive_measures']['long_term'] = ['Plant companion plants that repel aphids', 'Use reflective mulch', 'Monitor regularly']
        result['solutions'] = ['Apply neem oil spray', 'Use insecticidal soap', 'Introduce beneficial insects like ladybugs']
    elif 'wilt' in ai_lower or 'wilting' in ai_lower:
        result['problem_name'] = 'Plant Wilt Disease'
        result['description'] = 'Plant wilting due to root or vascular diseases. Often caused by fungal pathogens in the soil.'
        result['confidence'] = 82
        result['preventive_measures']['immediate'] = ['Remove affected plants', 'Improve drainage', 'Avoid overwatering']
        result['preventive_measures']['long_term'] = ['Use disease-free seeds', 'Practice crop rotation', 'Improve soil health']
        result['solutions'] = ['Apply fungicides to soil', 'Remove and destroy infected plants', 'Improve soil drainage']
    
    return result