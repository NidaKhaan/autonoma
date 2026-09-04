using UnityEngine;
using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;

public class SocketManager : MonoBehaviour
{
    public static SocketManager Instance;
    public string serverUrl = "ws://localhost:9090";

    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;
    private bool _isConnected = false;

    public ConcurrentQueue<string> ReceiveQueue = new ConcurrentQueue<string>();
    private ConcurrentQueue<string> _sendQueue  = new ConcurrentQueue<string>();

    void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
        _cts = new CancellationTokenSource();
        _ = ConnectAsync();
    }

    async Task ConnectAsync()
    {
        while (!_cts.Token.IsCancellationRequested)
        {
            try
            {
                _ws = new ClientWebSocket();
                Debug.Log("[Socket] Connecting to Python AI...");
                await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
                _isConnected = true;
                Debug.Log("[Socket] Connected!");
                await Task.WhenAny(SendLoop(), ReceiveLoop());
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Socket] Failed: {e.Message} — retrying in 3s");
                _isConnected = false;
            }
            await Task.Delay(3000);
        }
    }

    async Task SendLoop()
    {
        while (_ws.State == WebSocketState.Open)
        {
            if (_sendQueue.TryDequeue(out string msg))
            {
                var b = Encoding.UTF8.GetBytes(msg);
                await _ws.SendAsync(new ArraySegment<byte>(b),
                    WebSocketMessageType.Text, true, _cts.Token);
            }
            await Task.Delay(1);
        }
    }

    async Task ReceiveLoop()
    {
        var buffer = new byte[1024 * 64];
        while (_ws.State == WebSocketState.Open)
        {
            var result = await _ws.ReceiveAsync(
                new ArraySegment<byte>(buffer), _cts.Token);
            if (result.MessageType == WebSocketMessageType.Close) break;
            ReceiveQueue.Enqueue(
                Encoding.UTF8.GetString(buffer, 0, result.Count));
        }
        _isConnected = false;
    }

    public void Send(string json)
    {
        if (_isConnected) _sendQueue.Enqueue(json);
    }

    public bool IsConnected => _isConnected;

    void OnDestroy()
    {
        _cts?.Cancel();
        _ws?.Dispose();
    }
}