export async function onRequestGet(context) {
  // Simple admin endpoint to download all responses
  const secret = context.request.headers.get('x-admin-key');
  if (secret !== context.env.ADMIN_KEY) {
    return new Response('Unauthorized', { status: 401 });
  }
  
  const list = await context.env.SURVEY_KV.list({ prefix: 'response:' });
  const responses = [];
  
  for (const key of list.keys) {
    const val = await context.env.SURVEY_KV.get(key.name);
    if (val) responses.push(JSON.parse(val));
  }
  
  return new Response(JSON.stringify({ count: responses.length, responses }), {
    headers: { 'Content-Type': 'application/json' },
  });
}
