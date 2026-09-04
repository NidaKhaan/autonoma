using UnityEngine;

public class ObstacleAvoider : MonoBehaviour
{
    [Header("Detection Distances")]
    public float emergencyDist = 12f;
    public float dodgeDist     = 20f;
    public float slowDist      = 28f;

    public enum AvoidState { Clear, Slowing, Dodging, Emergency }
    public AvoidState State       { get; private set; } = AvoidState.Clear;
    public float      Throttle    { get; private set; } = 0.55f;
    public float      Brake       { get; private set; } = 0f;
    public float      SteerOffset { get; private set; } = 0f;
    public string     Reason      { get; private set; } = "Clear";

    private RaycastSensor _sensor;

    void Start()
    {
        _sensor = GetComponent<RaycastSensor>();
    }

    public void Evaluate(float wpSteer, float wpThrottle)
    {
        float fDist  = _sensor.frontDist;
        float flDist = _sensor.frontLeftDist;
        float frDist = _sensor.frontRightDist;
        float hlDist = _sensor.hardLeftDist;
        float hrDist = _sensor.hardRightDist;

        // ── EMERGENCY ─────────────────────────────────────────────────────
        if (fDist < emergencyDist)
        {
            State = AvoidState.Emergency;

            bool rightOpen = frDist > 3f && hrDist > 2f;
            bool leftOpen  = flDist > 3f && hlDist > 2f;

            // Brake power — jitna qareeb utna zyada
            float brakePow = Mathf.Lerp(0.5f, 1f,
                1f - (fDist / emergencyDist));

            if (rightOpen && (!leftOpen || frDist > flDist))
            {
                // Strong steer right + brake
                Throttle    = 0.4f;  // thoda throttle — reverse nahi
                Brake       = brakePow * 0.5f; // half brake taake move kare
                SteerOffset = 1f;    // full right
                Reason      = $"DODGE RIGHT — {fDist:F1}m";
            }
            else if (leftOpen)
            {
                Throttle    = 0.4f;
                Brake       = brakePow * 0.5f;
                SteerOffset = -1f;   // full left
                Reason      = $"DODGE LEFT — {fDist:F1}m";
            }
            else
            {
                // Dono sides blocked — full brake
                Throttle    = 0f;
                Brake       = 1f;
                SteerOffset = 0f;
                Reason      = $"FULL BRAKE — {fDist:F1}m";
            }
            return;
        }

        // ── DODGE ZONE ────────────────────────────────────────────────────
        if (fDist < dodgeDist)
        {
            State = AvoidState.Dodging;
            float urgency  = (dodgeDist - fDist) / dodgeDist; // 0 to 1
            bool canRight  = frDist > 3f && hrDist > 2f;
            bool canLeft   = flDist > 3f && hlDist > 2f;

            if (canRight && (!canLeft || frDist > flDist))
            {
                Throttle    = Mathf.Lerp(wpThrottle, 0.3f, urgency);
                Brake       = 0f;
                SteerOffset = Mathf.Lerp(0.3f, 0.9f, urgency); // early soft, late hard
                Reason      = $"AVOID RIGHT — {fDist:F1}m";
            }
            else if (canLeft)
            {
                Throttle    = Mathf.Lerp(wpThrottle, 0.3f, urgency);
                Brake       = 0f;
                SteerOffset = Mathf.Lerp(-0.3f, -0.9f, urgency);
                Reason      = $"AVOID LEFT — {fDist:F1}m";
            }
            else
            {
                Throttle    = Mathf.Lerp(wpThrottle, 0f, urgency);
                Brake       = urgency * 0.6f;
                SteerOffset = 0f;
                Reason      = $"SLOWING — {fDist:F1}m";
            }
            return;
        }

        // ── SLOW ZONE ─────────────────────────────────────────────────────
        if (fDist < slowDist)
        {
            State = AvoidState.Slowing;
            float urgency = (slowDist - fDist) / slowDist;
            bool earlyRight = frDist > flDist;

            Throttle    = wpThrottle * (1f - urgency * 0.3f);
            Brake       = 0f;
            SteerOffset = earlyRight
                ? Mathf.Lerp(0f, 0.3f, urgency)
                : Mathf.Lerp(0f, -0.3f, urgency);
            Reason      = $"CAUTION — {fDist:F1}m";
            return;
        }

        // ── CLEAR ─────────────────────────────────────────────────────────
        State       = AvoidState.Clear;
        Throttle    = wpThrottle;
        Brake       = 0f;
        SteerOffset = 0f;
        Reason      = "CLEAR";
    }
}