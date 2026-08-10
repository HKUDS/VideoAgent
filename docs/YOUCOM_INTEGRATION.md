# You.com Web Search Integration

This integration adds optional web search capabilities to VideoAgent using the You.com Search API. This enables VideoAgent to find current trends, reference materials, and contextual information for video content creation.

## Setup

### Environment Variables

Add these environment variables to enable web search:

```bash
# Required: You.com API key
export YOUCOM_API_KEY="your-api-key-here"

# Optional: Enable web search (default: false)  
export YOUCOM_SEARCH_ENABLED="true"

# Optional: Override API base URL (default: https://api.you.com)
export YOUCOM_BASE_URL="https://api.you.com"
```

### Getting a You.com API Key

1. Visit [You.com API](https://api.you.com) 
2. Sign up or log in to your account
3. Navigate to API settings and generate an API key
4. Set the `YOUCOM_API_KEY` environment variable

## Usage

Once configured, VideoAgent will automatically use web search when relevant to user requests. The system uses intent analysis to determine when web search would be helpful.

### Example Scenarios

Web search is particularly useful for:

- **Current Events**: "Create a video about the latest AI developments in 2026"
- **Trend Analysis**: "Make a video about popular TikTok trends this month" 
- **Research Content**: "Generate a video explaining recent climate change findings"
- **News Updates**: "Create a news-style video about recent tech company announcements"

### Manual Web Search

You can also explicitly request web search by using relevant keywords:

```
"Search for information about [topic] and create a video"
"Find current trends in [domain] and make educational content"
"Look up recent news about [subject] for video material"
```

## Integration Details

### New Tools Added

- **WebSearcher**: Core web search functionality with support for general web search and news-specific search

### New Intents

The following intents now trigger web search capability:

- `Web Search`: General web search requests
- `Current Information`: Finding up-to-date information  
- `Trend Research`: Researching current trends
- `News Search`: News-specific searches
- `Content Research`: Research for content creation

### API Endpoints Used

- `/search`: General web search results
- `/news`: News-specific search results

### Error Handling

The integration includes comprehensive error handling:

- **Disabled State**: When `YOUCOM_SEARCH_ENABLED` is false or API key is missing
- **Authentication Errors**: Invalid API key handling
- **Rate Limiting**: Graceful handling of API rate limits  
- **Network Errors**: Timeout and connection error handling
- **Validation**: Input parameter validation

### Security

- API keys are loaded from environment variables (not hardcoded)
- HTTPS-only communication with You.com API
- Request timeout limits prevent hanging
- No API keys logged in output

## Testing

### Basic Functionality Test

```python
from environment.roles.web_searcher import WebSearcher

# Test with minimal configuration
searcher = WebSearcher()
result = searcher.run(
    query="latest AI developments", 
    search_type="web",
    max_results=3
)
print(f"Status: {result.status}")
print(f"Results: {len(result.results)}")
```

### Integration Test

Run VideoAgent with a web search request:

```bash
python main.py
# When prompted, try: "Create a video about the latest technology trends in 2026"
```

## Fallback Behavior

- When web search is disabled: VideoAgent continues normal operation without web search
- When API key is invalid: Clear error messages, graceful degradation
- When API is unreachable: Timeout handling, user-friendly error messages
- When rate limited: Informative error message with retry guidance

## Troubleshooting

### Common Issues

1. **"Web search is disabled"**
   - Solution: Set `YOUCOM_SEARCH_ENABLED=true` and provide valid `YOUCOM_API_KEY`

2. **"Authentication failed"**  
   - Solution: Check that `YOUCOM_API_KEY` is valid and properly set

3. **"Rate limit exceeded"**
   - Solution: Wait before retrying, consider upgrading API plan if needed

4. **"Request timed out"**
   - Solution: Check internet connection, try again

### Debug Logging

Enable debug logging to see web search activity:

```python
import logging
logging.getLogger('environment.roles.web_searcher').setLevel(logging.DEBUG)
```

## Best Practices

- Keep search queries specific and relevant
- Use appropriate `search_type` ("web" vs "news") 
- Limit `max_results` to avoid information overload
- Monitor API usage to stay within rate limits
- Set reasonable timeout values for your use case