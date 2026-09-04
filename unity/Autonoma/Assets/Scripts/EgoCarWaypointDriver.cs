using UnityEngine;
using System.Collections.Generic;
using FCG;

[RequireComponent(typeof(PrometeoCarController))]
[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(RaycastSensor))]
public class EgoCarWaypointDriver : MonoBehaviour
{
    [Header("Waypoint Following")]
    public float waypointReachRadius = 4f;
    public float lookAheadDistance   = 8f;
    public float cruiseSpeed         = 25f;
    public float steerSensitivity    = 1.4f;

    [Header("Debug")]
    public bool drawPath = true;

    // Safety thresholds — hardcoded, do not expose to inspector
    // so nobody accidentally changes them
    private const float DIST_EMERGENCY = 4.5f;  // full brake instantly
    private const float DIST_HARD      = 10f;   // heavy brake ramp
    private const float DIST_SLOW      = 18f;   // ease off throttle

    // ── Public state for HUD ─────────────────────────────────────────────
    public string CurrentAction    { get; private set; } = "INIT";
    public float  CurrentRisk      { get; private set; } = 0f;
    public string CurrentRiskLevel { get; private set; } = "SAFE";
    public float  CurrentSpeed     { get; private set; } = 0f;
    public float  FinalThrottle    { get; private set; } = 0f;
    public float  FinalBrake       { get; private set; } = 0f;
    public float  FinalSteer       { get; private set; } = 0f;

    // Kept for AIDriver compatibility — not used for driving
    [HideInInspector] public float aiThrottle = 0f;
    [HideInInspector] public float aiBrake    = 0f;
    [HideInInspector] public float aiSteer    = 0f;
    [HideInInspector] public bool  aiOverride = false;

    private List<Vector3>         _path    = new List<Vector3>();
    private int                   _pathIdx = 0;
    private FCGWaypointsContainer _currentLane;
    private int                   _laneSide = 1;
    private PrometeoCarController _car;
    private Rigidbody             _rb;
    private RaycastSensor         _sensor;

    private float _stuckTimer  = 0f;
    private float _yellowTimer = 0f;
    private bool  _wasYellow   = false;

    void Start()
    {
        _car    = GetComponent<PrometeoCarController>();
        _rb     = GetComponent<Rigidbody>();
        _sensor = GetComponent<RaycastSensor>();
        FindNearestLane();
    }

    void Update()
    {
        CurrentSpeed = _rb.linearVelocity.magnitude * 3.6f;
        if (_path.Count == 0) { FindNearestLane(); return; }

        // ── Waypoint advance ─────────────────────────────────────────────
        Vector3 target = _path[_pathIdx];
        float dist2D = Vector3.Distance(
            new Vector3(transform.position.x, 0, transform.position.z),
            new Vector3(target.x, 0, target.z));
        if (dist2D < waypointReachRadius)
        {
            _pathIdx++;
            if (_pathIdx >= _path.Count) AdvanceToNextLane();
            return;
        }

        // ── Steering toward waypoint ─────────────────────────────────────
        Vector3 localTarget  = transform.InverseTransformPoint(target);
        float   wpSteer      = Mathf.Clamp(
            (localTarget.x / lookAheadDistance) * steerSensitivity, -1f, 1f);
        float   speedKmh     = Mathf.Abs(_car.carSpeed);
        float   wpThrottle   = speedKmh < cruiseSpeed ? 0.55f : 0f;

        // ── INSTANT front distance — read this frame, not last FixedUpdate ─
        float fDist  = _sensor.InstantFrontDist(22f);
        float flDist = _sensor.frontLeftDist;
        float frDist = _sensor.frontRightDist;
        float hlDist = _sensor.hardLeftDist;
        float hrDist = _sensor.hardRightDist;
        float llDist = _sensor.leftLaneDist;
        float rlDist = _sensor.rightLaneDist;

        // ── Stuck detection ──────────────────────────────────────────────
        if (_rb.linearVelocity.magnitude > 0.5f)
            _stuckTimer = 0f;
        else
        {
            _stuckTimer += Time.deltaTime;
            if (_stuckTimer > 4f)
            {
                _stuckTimer = 0f;
                Apply(0f, 0.6f, 0f, "UNSTUCK-BRAKE", 80f, "CRITICAL");
                return;
            }
        }

        // ════════════════════════════════════════════════════════════════
        // PRIORITY 1 — RED LIGHT
        // ════════════════════════════════════════════════════════════════
        if (_sensor.redLightAhead)
        {
            _yellowTimer = 0f; _wasYellow = false;
            Apply(0f, 1f, wpSteer, "RED LIGHT STOP", 90f, "CRITICAL");
            return;
        }

        // Yellow light
        if (_sensor.yellowLightAhead)
        {
            if (!_wasYellow) { _yellowTimer = 0f; _wasYellow = true; }
            _yellowTimer += Time.deltaTime;
            if (_yellowTimer < 2f)
            {
                Apply(wpThrottle * 0.3f, 0.3f, wpSteer, "YELLOW SLOW", 50f, "CAUTION");
                return;
            }
        }
        else { _wasYellow = false; _yellowTimer = 0f; }

        // ════════════════════════════════════════════════════════════════
        // PRIORITY 2 — EMERGENCY BRAKE  (< 4.5 m)
        // NO blending, NO smoothing — raw full brake immediately
        // ════════════════════════════════════════════════════════════════
        if (fDist < DIST_EMERGENCY)
        {
            _car.SetAIInputs(0f, 1f, wpSteer); // bypass Apply — direct call
            FinalThrottle  = 0f;
            FinalBrake     = 1f;
            FinalSteer     = wpSteer;
            CurrentAction  = $"!! EMERGENCY BRAKE {fDist:F1}m";
            CurrentRisk    = 100f;
            CurrentRiskLevel = "CRITICAL";
            return;
        }

        // ════════════════════════════════════════════════════════════════
        // PRIORITY 3 — HARD BRAKE  (4.5 m – 10 m)
        // Linear ramp: closer = more brake
        // ════════════════════════════════════════════════════════════════
        if (fDist < DIST_HARD)
        {
            float t = 1f - ((fDist - DIST_EMERGENCY) /
                            (DIST_HARD - DIST_EMERGENCY));
            t = Mathf.Clamp01(t);

            // brake 0.4 at edge of zone → 0.95 at emergency boundary
            float brakePower = Mathf.Lerp(0.4f, 0.95f, t);

            Apply(0f, brakePower, wpSteer,
                  $"HARD BRAKE {fDist:F1}m",
                  60f + t * 38f, "HIGH");
            return;
        }

        // ════════════════════════════════════════════════════════════════
        // PRIORITY 4 — SLOW ZONE  (10 m – 18 m)
        // Only reduce throttle, tiny brake
        // ════════════════════════════════════════════════════════════════
        if (fDist < DIST_SLOW)
        {
            float u = 1f - ((fDist - DIST_HARD) /
                            (DIST_SLOW - DIST_HARD));
            u = Mathf.Clamp01(u);

            float slowThrottle = wpThrottle * (1f - u * 0.8f);
            float slowBrake    = u * 0.12f;

            Apply(slowThrottle, slowBrake, wpSteer,
                  $"SLOWING {fDist:F1}m", u * 60f, "CAUTION");
            return;
        }

        // ════════════════════════════════════════════════════════════════
        // PRIORITY 5 — LANE KEEPING
        // ════════════════════════════════════════════════════════════════
        if (llDist < 2.5f) wpSteer = Mathf.Max(wpSteer,  0.3f);
        if (rlDist < 2.5f) wpSteer = Mathf.Min(wpSteer, -0.3f);
        if (hlDist < 3f)   wpSteer = Mathf.Max(wpSteer,  0.2f);
        if (hrDist < 3f)   wpSteer = Mathf.Min(wpSteer, -0.2f);

        // ════════════════════════════════════════════════════════════════
        // DEFAULT — CRUISE
        // ════════════════════════════════════════════════════════════════
        Apply(wpThrottle, 0f, wpSteer, $"CRUISE {fDist:F0}m", 0f, "SAFE");
    }

    void Apply(float t, float b, float s,
               string action, float risk, string lvl)
    {
        if (t > 0.05f && b > 0.05f) t = 0f;
        FinalThrottle    = t;
        FinalBrake       = b;
        FinalSteer       = s;
        CurrentAction    = action;
        CurrentRisk      = risk;
        CurrentRiskLevel = lvl;
        _car.SetAIInputs(t, b, s);

        if (drawPath && _path.Count > _pathIdx + 1)
        {
            for (int i = _pathIdx; i < _path.Count - 1; i++)
                Debug.DrawLine(_path[i] + Vector3.up * 0.5f,
                               _path[i+1] + Vector3.up * 0.5f, Color.green);
            Debug.DrawLine(transform.position + Vector3.up * 0.5f,
                           _path[_pathIdx] + Vector3.up * 0.5f, Color.yellow);
        }
    }

    void FindNearestLane()
    {
        var allLanes = FindObjectsOfType<FCGWaypointsContainer>();
        float bestDist = float.MaxValue;
        FCGWaypointsContainer bestLane = null;
        int bestSide = 1, bestIdx = 0;

        foreach (var lane in allLanes)
        {
            if (lane.waypoints == null || lane.waypoints.Count < 2) continue;
            if (lane.bloked) continue;
            for (int side = 0; side <= 1; side++)
            {
                if (lane.oneway && !lane.doubleLine && side == 0) continue;
                for (int i = 0; i < lane.waypoints.Count; i++)
                {
                    float d = Vector3.Distance(transform.position, lane.Node(side, i));
                    if (d < bestDist)
                    { bestDist = d; bestLane = lane; bestSide = side; bestIdx = i; }
                }
            }
        }

        if (bestLane == null) { Debug.LogWarning("[WPDriver] No lane!"); return; }
        _currentLane = bestLane;
        _laneSide    = bestSide;
        BuildPathFromLane(_currentLane, _laneSide, bestIdx);
    }

    void BuildPathFromLane(FCGWaypointsContainer lane, int side, int startIdx)
    {
        _path.Clear(); _pathIdx = 0;
        int total = lane.GetTotalNodes();
        for (int i = startIdx; i <= total; i++)
            _path.Add(lane.Node(side, i));
    }

    void AdvanceToNextLane()
    {
        var nexts     = _laneSide == 1 ? _currentLane.nextWay1     : _currentLane.nextWay0;
        var nextSides = _laneSide == 1 ? _currentLane.nextWaySide1 : _currentLane.nextWaySide0;
        if (nexts == null || nexts.Length == 0) { FindNearestLane(); return; }
        int pick     = Random.Range(0, nexts.Length);
        _currentLane = nexts[pick];
        _laneSide    = nextSides[pick];
        BuildPathFromLane(_currentLane, _laneSide, 0);
    }

    void OnCollisionEnter(Collision col)
    {
        Apply(0f, 1f, 0f, "COLLISION BRAKE", 100f, "CRITICAL");
    }
}