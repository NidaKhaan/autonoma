using UnityEngine;
using System.Collections;

[RequireComponent(typeof(RaycastSensor))]
[RequireComponent(typeof(PrometeoCarController))]
[RequireComponent(typeof(EgoCarWaypointDriver))]
public class AIDriver : MonoBehaviour
{
    [Header("Camera")]
    public Camera frontCamera;

    [Header("Collision Tracking")]
    public int collisionCars  = 0;
    public int collisionPeds  = 0;
    public int collisionOther = 0;

    private RaycastSensor         _sensor;
    private Rigidbody             _rb;
    private EgoCarWaypointDriver  _wpDriver;

    // Read-only properties for UI/HUD
    public float  Throttle  => _wpDriver.FinalThrottle;
    public float  Brake     => _wpDriver.FinalBrake;
    public float  Steer     => _wpDriver.FinalSteer;
    public string Action    => _wpDriver.CurrentAction;
    public float  Risk      => _wpDriver.CurrentRisk;
    public string RiskLevel => _wpDriver.CurrentRiskLevel;

    void Start()
    {
        _sensor   = GetComponent<RaycastSensor>();
        _rb       = GetComponent<Rigidbody>();
        _wpDriver = GetComponent<EgoCarWaypointDriver>();

        // Make sure waypoint driver is always in control
        _wpDriver.aiOverride = false;

        StartCoroutine(SendFrameLoop());
        StartCoroutine(ReadResponseLoop());
    }

    IEnumerator SendFrameLoop()
    {
        var wait = new WaitForSeconds(0.1f);
        while (true)
        {
            yield return wait;
            if (SocketManager.Instance == null || !SocketManager.Instance.IsConnected) continue;

            string b64 = "";
            if (frontCamera != null)
            {
                RenderTexture rt = new RenderTexture(320, 240, 24);
                frontCamera.targetTexture = rt;
                frontCamera.Render();
                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(320, 240, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, 320, 240), 0, 0);
                tex.Apply();
                frontCamera.targetTexture = null;
                RenderTexture.active = null;
                Destroy(rt);
                b64 = System.Convert.ToBase64String(tex.EncodeToJPG(60));
                Destroy(tex);
            }

            var ci = System.Globalization.CultureInfo.InvariantCulture;
            string stateJson = string.Format(ci,
                @"{{""speed_ms"":{0:F4},""front_dist"":{1:F4},""front_left_dist"":{2:F4}," +
                @"""front_right_dist"":{3:F4},""hard_left_dist"":{4:F4},""hard_right_dist"":{5:F4}," +
                @"""left_lane_dist"":{6:F4},""right_lane_dist"":{7:F4},""rear_dist"":{8:F4}," +
                @"""red_light"":{9},""pos_x"":{10:F4},""pos_y"":{11:F4}," +
                @"""col_cars"":{12},""col_peds"":{13},""col_other"":{14}}}",
                _rb.linearVelocity.magnitude,
                _sensor.frontDist, _sensor.frontLeftDist, _sensor.frontRightDist,
                _sensor.hardLeftDist, _sensor.hardRightDist,
                _sensor.leftLaneDist, _sensor.rightLaneDist, _sensor.backDist,
                _sensor.redLightAhead ? "true" : "false",
                transform.position.x, transform.position.z,
                collisionCars, collisionPeds, collisionOther
            );

            string fullJson = "{\"frame\":\"" + b64 + "\",\"state\":" + stateJson + "}";
            SocketManager.Instance.Send(fullJson);
        }
    }

    IEnumerator ReadResponseLoop()
    {
        while (true)
        {
            yield return null;
            if (SocketManager.Instance == null) continue;
            if (!SocketManager.Instance.ReceiveQueue.TryDequeue(out string json)) continue;

            // AIDriver no longer controls the car.
            // Socket data is only logged/stored — EgoCarWaypointDriver drives.
            // You can log Python's suggestion here if needed for debugging:
            // Debug.Log("[AIDriver] Python says: " + json);
        }
    }

    float ParseFloat(string json, string key)
    {
        string search = "\"" + key + "\":";
        int idx = json.IndexOf(search);
        if (idx < 0) return 0f;
        int start = idx + search.Length;
        while (start < json.Length && json[start] == ' ') start++;
        int end = start;
        while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '.' || json[end] == '-')) end++;
        return float.Parse(json.Substring(start, end - start),
            System.Globalization.CultureInfo.InvariantCulture);
    }

    string ParseString(string json, string key)
    {
        string search = "\"" + key + "\":\"";
        int idx = json.IndexOf(search);
        if (idx < 0) return "";
        int start = idx + search.Length;
        int end = json.IndexOf("\"", start);
        return end < 0 ? "" : json.Substring(start, end - start);
    }

    void OnCollisionEnter(Collision col)
    {
        if      (col.gameObject.CompareTag("Car"))    collisionCars++;
        else if (col.gameObject.CompareTag("Person")) collisionPeds++;
        else                                          collisionOther++;
    }
}