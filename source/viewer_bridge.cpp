/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "viewer_bridge.h"

#include <iomanip>
#include <sstream>
#include <string>

#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace
{
float elapsedMs(std::chrono::high_resolution_clock::time_point begin,
                std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<float, std::milli>(end - begin).count();
}

SOCKET asSocket(uintptr_t value)
{
    return (SOCKET)value;
}

std::string trim(const std::string &value)
{
    size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return "";
    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string lowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
                   { return (char)std::tolower(c); });
    return value;
}

std::string headerValue(const std::string &request, const char *headerName)
{
    std::string header = lowerAscii(headerName);
    size_t lineBegin = 0;
    while (lineBegin < request.size())
    {
        size_t lineEnd = request.find("\r\n", lineBegin);
        if (lineEnd == std::string::npos)
            lineEnd = request.size();
        std::string line = request.substr(lineBegin, lineEnd - lineBegin);
        size_t colon = line.find(':');
        if (colon != std::string::npos && lowerAscii(trim(line.substr(0, colon))) == header)
            return trim(line.substr(colon + 1));
        lineBegin = lineEnd + 2;
    }
    return "";
}

uint32_t rol(uint32_t value, int bits)
{
    return (value << bits) | (value >> (32 - bits));
}

std::array<uint8_t, 20> sha1(const std::string &input)
{
    uint64_t bitLength = (uint64_t)input.size() * 8ull;
    std::vector<uint8_t> data(input.begin(), input.end());
    data.push_back(0x80);
    while ((data.size() % 64) != 56)
        data.push_back(0);
    for (int i = 7; i >= 0; --i)
        data.push_back((uint8_t)((bitLength >> (i * 8)) & 0xff));

    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xc3d2e1f0;

    for (size_t offset = 0; offset < data.size(); offset += 64)
    {
        uint32_t w[80] = {};
        for (int i = 0; i < 16; ++i)
        {
            size_t j = offset + i * 4;
            w[i] = ((uint32_t)data[j] << 24) |
                   ((uint32_t)data[j + 1] << 16) |
                   ((uint32_t)data[j + 2] << 8) |
                   (uint32_t)data[j + 3];
        }
        for (int i = 16; i < 80; ++i)
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;
        for (int i = 0; i < 80; ++i)
        {
            uint32_t f = 0;
            uint32_t k = 0;
            if (i < 20)
            {
                f = (b & c) | ((~b) & d);
                k = 0x5a827999;
            }
            else if (i < 40)
            {
                f = b ^ c ^ d;
                k = 0x6ed9eba1;
            }
            else if (i < 60)
            {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8f1bbcdc;
            }
            else
            {
                f = b ^ c ^ d;
                k = 0xca62c1d6;
            }
            uint32_t temp = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = temp;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<uint8_t, 20> digest = {};
    uint32_t h[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i)
    {
        digest[i * 4 + 0] = (uint8_t)((h[i] >> 24) & 0xff);
        digest[i * 4 + 1] = (uint8_t)((h[i] >> 16) & 0xff);
        digest[i * 4 + 2] = (uint8_t)((h[i] >> 8) & 0xff);
        digest[i * 4 + 3] = (uint8_t)(h[i] & 0xff);
    }
    return digest;
}

std::string base64(const uint8_t *data, size_t size)
{
    static const char *table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (size_t i = 0; i < size; i += 3)
    {
        uint32_t value = (uint32_t)data[i] << 16;
        if (i + 1 < size)
            value |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < size)
            value |= (uint32_t)data[i + 2];
        out.push_back(table[(value >> 18) & 63]);
        out.push_back(table[(value >> 12) & 63]);
        out.push_back(i + 1 < size ? table[(value >> 6) & 63] : '=');
        out.push_back(i + 2 < size ? table[value & 63] : '=');
    }
    return out;
}

std::string webSocketAcceptKey(const std::string &key)
{
    std::string combined = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::array<uint8_t, 20> digest = sha1(combined);
    return base64(digest.data(), digest.size());
}

bool sendAll(SOCKET socket, const uint8_t *data, size_t size)
{
    size_t sent = 0;
    while (sent < size)
    {
        int chunk = send(socket, (const char *)data + sent, (int)std::min<size_t>(size - sent, 16384), 0);
        if (chunk == SOCKET_ERROR)
        {
            int error = WSAGetLastError();
            if (error == WSAEWOULDBLOCK)
            {
                fd_set writeSet;
                FD_ZERO(&writeSet);
                FD_SET(socket, &writeSet);
                timeval timeout = {};
                timeout.tv_sec = 1;
                int ready = select(0, 0, &writeSet, 0, &timeout);
                if (ready > 0)
                    continue;
            }
            return false;
        }
        if (chunk <= 0)
            return false;
        sent += (size_t)chunk;
    }
    return true;
}

bool sendTextFrame(SOCKET socket, const std::string &payload)
{
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 16);
    frame.push_back(0x81);
    if (payload.size() < 126)
    {
        frame.push_back((uint8_t)payload.size());
    }
    else if (payload.size() <= 0xffff)
    {
        frame.push_back(126);
        frame.push_back((uint8_t)((payload.size() >> 8) & 0xff));
        frame.push_back((uint8_t)(payload.size() & 0xff));
    }
    else
    {
        frame.push_back(127);
        uint64_t size = (uint64_t)payload.size();
        for (int i = 7; i >= 0; --i)
            frame.push_back((uint8_t)((size >> (i * 8)) & 0xff));
    }
    frame.insert(frame.end(), payload.begin(), payload.end());
    return sendAll(socket, frame.data(), frame.size());
}

bool sendBinaryFrame(SOCKET socket, const std::vector<uint8_t> &payload)
{
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 16);
    frame.push_back(0x82);
    if (payload.size() < 126)
    {
        frame.push_back((uint8_t)payload.size());
    }
    else if (payload.size() <= 0xffff)
    {
        frame.push_back(126);
        frame.push_back((uint8_t)((payload.size() >> 8) & 0xff));
        frame.push_back((uint8_t)(payload.size() & 0xff));
    }
    else
    {
        frame.push_back(127);
        uint64_t size = (uint64_t)payload.size();
        for (int i = 7; i >= 0; --i)
            frame.push_back((uint8_t)((size >> (i * 8)) & 0xff));
    }
    frame.insert(frame.end(), payload.begin(), payload.end());
    return sendAll(socket, frame.data(), frame.size());
}

bool receiveTextFrame(SOCKET socket, std::string &payload, bool &closed)
{
    closed = false;
    uint8_t header[2] = {};
    int received = recv(socket, (char *)header, 2, 0);
    if (received == SOCKET_ERROR)
    {
        int error = WSAGetLastError();
        if (error == WSAEWOULDBLOCK)
            return false;
        closed = true;
        return false;
    }
    if (received == 0)
    {
        closed = true;
        return false;
    }
    if (received != 2)
    {
        closed = true;
        return false;
    }

    uint8_t opcode = header[0] & 0x0f;
    bool masked = (header[1] & 0x80) != 0;
    uint64_t length = header[1] & 0x7f;
    if (length == 126)
    {
        uint8_t ext[2] = {};
        if (recv(socket, (char *)ext, 2, 0) != 2)
        {
            closed = true;
            return false;
        }
        length = ((uint64_t)ext[0] << 8) | ext[1];
    }
    else if (length == 127)
    {
        uint8_t ext[8] = {};
        if (recv(socket, (char *)ext, 8, 0) != 8)
        {
            closed = true;
            return false;
        }
        length = 0;
        for (int i = 0; i < 8; ++i)
            length = (length << 8) | ext[i];
    }

    if (length > 65536)
    {
        closed = true;
        return false;
    }

    uint8_t mask[4] = {};
    if (masked && recv(socket, (char *)mask, 4, 0) != 4)
    {
        closed = true;
        return false;
    }

    std::vector<uint8_t> data((size_t)length);
    size_t offset = 0;
    while (offset < data.size())
    {
        int chunk = recv(socket, (char *)data.data() + offset, (int)(data.size() - offset), 0);
        if (chunk <= 0)
        {
            closed = true;
            return false;
        }
        offset += (size_t)chunk;
    }

    if (opcode == 0x8)
    {
        closed = true;
        return false;
    }
    if (opcode != 0x1)
        return false;

    if (masked)
    {
        for (size_t i = 0; i < data.size(); ++i)
            data[i] ^= mask[i & 3];
    }
    payload.assign(data.begin(), data.end());
    return true;
}

bool performHandshake(SOCKET client)
{
    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192)
    {
        int received = recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0)
            return false;
        request.append(buffer, buffer + received);
    }

    std::string key = headerValue(request, "Sec-WebSocket-Key");
    if (key.empty())
        return false;

    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " +
        webSocketAcceptKey(key) + "\r\n"
                                  "\r\n";
    return sendAll(client, (const uint8_t *)response.data(), response.size());
}

const char *shapeName(RigidShapeType type)
{
    switch (type)
    {
    case RIGID_SHAPE_BOX:
        return "box";
    case RIGID_SHAPE_SPHERE:
        return "sphere";
    case RIGID_SHAPE_CAPSULE:
        return "capsule";
    case RIGID_SHAPE_CYLINDER:
        return "cylinder";
    default:
        return "box";
    }
}

int materialIdForBody(const SimBodyData &body)
{
    if (body.mass <= 0.0f)
        return 0;

    switch (body.shape.type)
    {
    case RIGID_SHAPE_BOX:
        return 1;
    case RIGID_SHAPE_SPHERE:
        return 3;
    case RIGID_SHAPE_CYLINDER:
        return 4;
    case RIGID_SHAPE_CAPSULE:
        return 5;
    default:
        return 1;
    }
}

std::string jsonEscape(const char *text)
{
    std::string out;
    if (!text)
        return out;
    for (const char *c = text; *c; ++c)
    {
        if (*c == '"' || *c == '\\')
        {
            out.push_back('\\');
            out.push_back(*c);
        }
        else if (*c == '\n')
        {
            out += "\\n";
        }
        else if (*c == '\r')
        {
            out += "\\r";
        }
        else if (*c == '\t')
        {
            out += "\\t";
        }
        else if ((unsigned char)*c < 0x20)
        {
            char escaped[7];
            snprintf(escaped, sizeof(escaped), "\\u%04x", (unsigned char)*c);
            out += escaped;
        }
        else
        {
            out.push_back(*c);
        }
    }
    return out;
}

bool jsonStringField(const std::string &json, const char *name, std::string &value)
{
    std::string key = std::string("\"") + name + "\"";
    size_t keyPos = json.find(key);
    if (keyPos == std::string::npos)
        return false;
    size_t colon = json.find(':', keyPos + key.size());
    if (colon == std::string::npos)
        return false;
    size_t begin = json.find_first_not_of(" \t\r\n", colon + 1);
    if (begin == std::string::npos || json[begin] != '"')
        return false;
    std::string out;
    bool escape = false;
    for (size_t i = begin + 1; i < json.size(); ++i)
    {
        char c = json[i];
        if (escape)
        {
            out.push_back(c);
            escape = false;
        }
        else if (c == '\\')
        {
            escape = true;
        }
        else if (c == '"')
        {
            value = out;
            return true;
        }
        else
        {
            out.push_back(c);
        }
    }
    return false;
}

bool jsonBoolField(const std::string &json, const char *name, bool &value)
{
    std::string key = std::string("\"") + name + "\"";
    size_t keyPos = json.find(key);
    if (keyPos == std::string::npos)
        return false;
    size_t colon = json.find(':', keyPos + key.size());
    if (colon == std::string::npos)
        return false;
    size_t begin = json.find_first_not_of(" \t\r\n", colon + 1);
    if (begin == std::string::npos)
        return false;
    if (json.compare(begin, 4, "true") == 0)
    {
        value = true;
        return true;
    }
    if (json.compare(begin, 5, "false") == 0)
    {
        value = false;
        return true;
    }
    return false;
}

bool jsonNumberField(const std::string &json, const char *name, double &value)
{
    std::string key = std::string("\"") + name + "\"";
    size_t keyPos = json.find(key);
    if (keyPos == std::string::npos)
        return false;
    size_t colon = json.find(':', keyPos + key.size());
    if (colon == std::string::npos)
        return false;
    size_t begin = json.find_first_not_of(" \t\r\n", colon + 1);
    if (begin == std::string::npos)
        return false;
    char *end = 0;
    value = strtod(json.c_str() + begin, &end);
    return end != json.c_str() + begin;
}

bool jsonFloat3Field(const std::string &json, const char *name, float3 &value)
{
    std::string key = std::string("\"") + name + "\"";
    size_t keyPos = json.find(key);
    if (keyPos == std::string::npos)
        return false;
    size_t colon = json.find(':', keyPos + key.size());
    if (colon == std::string::npos)
        return false;
    size_t begin = json.find('[', colon + 1);
    if (begin == std::string::npos)
        return false;

    float values[3] = {0, 0, 0};
    const char *cursor = json.c_str() + begin + 1;
    for (int i = 0; i < 3; ++i)
    {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n' || *cursor == ',')
            ++cursor;
        char *end = 0;
        double parsed = strtod(cursor, &end);
        if (end == cursor)
            return false;
        values[i] = (float)parsed;
        cursor = end;
    }

    value = {values[0], values[1], values[2]};
    return true;
}

// Parses a JSON number array field of arbitrary length.
bool jsonNumberArrayField(const std::string &json, const char *name, std::vector<float> &out)
{
    std::string key = std::string("\"") + name + "\"";
    size_t keyPos = json.find(key);
    if (keyPos == std::string::npos)
        return false;
    size_t colon = json.find(':', keyPos + key.size());
    if (colon == std::string::npos)
        return false;
    size_t begin = json.find('[', colon + 1);
    if (begin == std::string::npos)
        return false;

    out.clear();
    const char *cursor = json.c_str() + begin + 1;
    for (;;)
    {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n' || *cursor == ',')
            ++cursor;
        if (*cursor == ']' || *cursor == '\0')
            break;
        char *end = 0;
        double parsed = strtod(cursor, &end);
        if (end == cursor)
            return false;
        out.push_back((float)parsed);
        cursor = end;
    }
    return true;
}

bool parseCommandJson(const std::string &json, SimulationCommand &command)
{
    std::string type;
    if (!jsonStringField(json, "type", type) || lowerAscii(type) != "command")
        return false;
    if (!jsonStringField(json, "command", command.command) || command.command.empty())
        return false;

    if (lowerAscii(command.command) == "importmesh")
    {
        jsonStringField(json, "name", command.meshName);
        jsonStringField(json, "mode", command.meshMode);
        std::vector<float> tris;
        if (!jsonNumberArrayField(json, "vertices", command.meshVertices) ||
            !jsonNumberArrayField(json, "triangles", tris))
            return false;
        command.meshTriangles.reserve(tris.size());
        for (float t : tris)
            command.meshTriangles.push_back((uint32_t)t);
        double spacing = 0.0, scale = 0.0;
        if (jsonNumberField(json, "spacing", spacing))
            command.meshSpacing = (float)spacing;
        if (jsonNumberField(json, "scale", scale) && scale > 0.0)
            command.meshScale = (float)scale;
        if (jsonFloat3Field(json, "position", command.meshPosition))
            command.hasMeshPosition = true;
        return true;
    }

    if (jsonStringField(json, "value", command.stringValue))
        return true;

    bool boolValue = false;
    if (jsonBoolField(json, "value", boolValue))
    {
        command.boolValue = boolValue;
        command.hasBool = true;
        return true;
    }

    double numberValue = 0.0;
    if (jsonNumberField(json, "value", numberValue))
    {
        command.numberValue = numberValue;
        command.hasNumber = true;
    }

    double bodyId = 0.0;
    if (jsonNumberField(json, "bodyId", bodyId))
    {
        command.bodyId = (uint32_t)bodyId;
        command.hasBodyId = true;
    }

    if (jsonFloat3Field(json, "localHit", command.localHit))
        command.hasLocalHit = true;
    if (jsonFloat3Field(json, "worldHit", command.worldHit))
        command.hasWorldHit = true;
    if (jsonFloat3Field(json, "worldTarget", command.worldTarget))
        command.hasWorldTarget = true;

    return true;
}

std::string makeStatusJson(const char *metricsText)
{
    std::ostringstream out;
    out << "{\"type\":\"status\",\"metrics\":\"" << jsonEscape(metricsText ? metricsText : "") << "\"}";
    return out.str();
}

std::string makeSnapshotJson(const SimWorld &world, const char *sceneName, uint64_t frame)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(5);
    out << "{\"type\":\"snapshot\",\"version\":1,\"frame\":" << frame
        << ",\"scene\":\"" << jsonEscape(sceneName) << "\",\"bodies\":[";
    bool first = true;
    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        if (!body.active)
            continue;
        if (!first)
            out << ",";
        first = false;
        out << "{\"id\":" << i
            << ",\"shape\":\"" << shapeName(body.shape.type) << "\""
            << ",\"position\":[" << body.positionLin.x << "," << body.positionLin.y << "," << body.positionLin.z << "]"
            << ",\"rotation\":[" << body.positionAng.x << "," << body.positionAng.y << "," << body.positionAng.z << "," << body.positionAng.w << "]"
            << ",\"size\":[" << body.shape.size.x << "," << body.shape.size.y << "," << body.shape.size.z << "]"
            << ",\"radius\":" << body.shape.radius
            << ",\"halfLength\":" << body.shape.halfLength
            << ",\"dynamic\":" << (body.mass > 0.0f ? "true" : "false")
            << ",\"material\":" << materialIdForBody(body)
            << "}";
    }
    out << "]}";
    return out.str();
}

template <typename T>
void appendPod(std::vector<uint8_t> &out, const T &value)
{
    const uint8_t *bytes = (const uint8_t *)&value;
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

void appendFloat3(std::vector<uint8_t> &out, const float3 &value)
{
    appendPod(out, value.x);
    appendPod(out, value.y);
    appendPod(out, value.z);
}

void appendQuat(std::vector<uint8_t> &out, const quat &value)
{
    appendPod(out, value.x);
    appendPod(out, value.y);
    appendPod(out, value.z);
    appendPod(out, value.w);
}

std::vector<uint8_t> makeSnapshotBinary(const SimWorld &world, const char *sceneName, uint64_t frame)
{
    std::string scene = sceneName ? sceneName : "";
    uint32_t activeBodies = 0;
    for (const SimBodyData &body : world.bodies)
    {
        if (body.active)
            activeBodies++;
    }

    std::vector<uint8_t> out;
    out.reserve(32 + scene.size() + (size_t)activeBodies * 64);
    const uint32_t magic = 0x53425641u; // AVBS, little-endian.
    const uint32_t version = 1;
    const uint32_t sceneBytes = (uint32_t)scene.size();
    appendPod(out, magic);
    appendPod(out, version);
    appendPod(out, frame);
    appendPod(out, sceneBytes);
    appendPod(out, activeBodies);
    out.insert(out.end(), scene.begin(), scene.end());

    for (size_t i = 0; i < world.bodies.size(); ++i)
    {
        const SimBodyData &body = world.bodies[i];
        if (!body.active)
            continue;
        uint32_t id = (uint32_t)i;
        uint32_t shape = (uint32_t)body.shape.type;
        uint32_t material = (uint32_t)materialIdForBody(body);
        uint32_t dynamic = body.mass > 0.0f ? 1u : 0u;
        appendPod(out, id);
        appendPod(out, shape);
        appendPod(out, material);
        appendPod(out, dynamic);
        appendFloat3(out, body.positionLin);
        appendQuat(out, body.positionAng);
        appendFloat3(out, body.shape.size);
        appendPod(out, body.shape.radius);
        appendPod(out, body.shape.halfLength);
    }
    return out;
}

bool handleBridgeLocalCommand(const SimulationCommand &command, bool &binarySnapshots)
{
    std::string name = lowerAscii(command.command);
    if (name != "snapshotmode" && name != "setsnapshotmode" && name != "binarysnapshots")
        return false;

    if (!command.stringValue.empty())
    {
        std::string mode = lowerAscii(command.stringValue);
        binarySnapshots = mode == "binary" || mode == "bin" || mode == "1" || mode == "true";
        return true;
    }
    if (command.hasBool)
    {
        binarySnapshots = command.boolValue;
        return true;
    }
    return false;
}
}
#endif

ViewerBridge::ViewerBridge()
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    running = false;
    clientCount = 0;
    listenSocket = (uintptr_t)INVALID_SOCKET;
    winsockStarted = false;
    metrics = ViewerBridgeStats{};
    strcpy_s(status, "Viewer bridge stopped");
#endif
}

ViewerBridge::~ViewerBridge()
{
    stop();
}

bool ViewerBridge::start(uint16_t port)
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    if (running)
        return true;
    setStatusText("Viewer bridge starting");
    running = true;
    serverThread = std::thread(&ViewerBridge::serverLoop, this, port);
    return true;
#else
    (void)port;
    return false;
#endif
}

void ViewerBridge::stop()
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    if (!running)
        return;
    running = false;
    SOCKET listener = asSocket(listenSocket);
    if (listener != INVALID_SOCKET)
        closesocket(listener);
    if (serverThread.joinable())
        serverThread.join();

    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        for (const ViewerBridgeClient &client : clients)
            closesocket(asSocket(client.socket));
        clients.clear();
    }
    clientCount = 0;
    listenSocket = (uintptr_t)INVALID_SOCKET;
    if (winsockStarted)
    {
        WSACleanup();
        winsockStarted = false;
    }
    setStatusText("Viewer bridge stopped");
#endif
}

void ViewerBridge::broadcastSnapshot(const SimWorld &world, const char *sceneName, uint64_t frame)
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    if (!running || clientCount <= 0)
        return;

    auto serializeBegin = std::chrono::high_resolution_clock::now();
    bool needJson = false;
    bool needBinary = false;
    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        for (const ViewerBridgeClient &client : clients)
        {
            if (client.binarySnapshots)
                needBinary = true;
            else
                needJson = true;
        }
    }
    std::string json;
    std::vector<uint8_t> binary;
    if (needBinary)
        binary = makeSnapshotBinary(world, sceneName, frame);
    if (needJson)
        json = makeSnapshotJson(world, sceneName, frame);
    float serializeMs = elapsedMs(serializeBegin, std::chrono::high_resolution_clock::now());

    auto sendBegin = std::chrono::high_resolution_clock::now();
    int sentClients = 0;
    int failures = 0;
    std::lock_guard<std::mutex> lock(clientsMutex);
    for (size_t i = 0; i < clients.size();)
    {
        SOCKET client = asSocket(clients[i].socket);
        bool sent = clients[i].binarySnapshots ? sendBinaryFrame(client, binary) : sendTextFrame(client, json);
        if (sent)
        {
            sentClients++;
            ++i;
        }
        else
        {
            failures++;
            closesocket(client);
            clients.erase(clients.begin() + i);
        }
    }
    clientCount = (int)clients.size();
    float sendMs = elapsedMs(sendBegin, std::chrono::high_resolution_clock::now());
    uint64_t reportedBytes = needBinary && !needJson ? (uint64_t)binary.size() : (uint64_t)json.size();
    updateSnapshotStats(reportedBytes,
                        serializeMs, sendMs, sentClients, failures);
#else
    (void)world;
    (void)sceneName;
    (void)frame;
#endif
}

void ViewerBridge::broadcastStatus(const char *metricsText)
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    if (!running || clientCount <= 0)
        return;

    std::string json = makeStatusJson(metricsText);
    std::lock_guard<std::mutex> lock(clientsMutex);
    for (size_t i = 0; i < clients.size();)
    {
        SOCKET client = asSocket(clients[i].socket);
        if (sendTextFrame(client, json))
        {
            ++i;
        }
        else
        {
            closesocket(client);
            clients.erase(clients.begin() + i);
        }
    }
    clientCount = (int)clients.size();
    updateStatusStats();
#else
    (void)metricsText;
#endif
}

void ViewerBridge::broadcastText(const std::string &json)
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    if (!running || clientCount <= 0)
        return;

    std::lock_guard<std::mutex> lock(clientsMutex);
    for (size_t i = 0; i < clients.size();)
    {
        SOCKET client = asSocket(clients[i].socket);
        if (sendTextFrame(client, json))
        {
            ++i;
        }
        else
        {
            closesocket(client);
            clients.erase(clients.begin() + i);
        }
    }
    clientCount = (int)clients.size();
#else
    (void)json;
#endif
}

bool ViewerBridge::pollCommand(SimulationCommand &command)
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (commands.empty())
        return false;
    command = commands.front();
    commands.pop_front();
    return true;
#else
    (void)command;
    return false;
#endif
}

const char *ViewerBridge::statusText() const
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    return status;
#else
    return "Viewer bridge disabled";
#endif
}

int ViewerBridge::clientCountValue() const
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    return clientCount;
#else
    return 0;
#endif
}

ViewerBridgeStats ViewerBridge::statsSnapshot() const
{
#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
    ViewerBridgeStats copy;
    {
        std::lock_guard<std::mutex> metricsLock(metricsMutex);
        copy = metrics;
    }
    copy.clients = clientCount;
    copy.binarySnapshotMode = 0;
    {
        std::lock_guard<std::mutex> clientsLock(clientsMutex);
        for (const ViewerBridgeClient &client : clients)
        {
            if (client.binarySnapshots)
            {
                copy.binarySnapshotMode = 1;
                break;
            }
        }
    }
    {
        std::lock_guard<std::mutex> commandLock(commandsMutex);
        copy.queuedCommands = (int)commands.size();
    }
    return copy;
#else
    return ViewerBridgeStats{};
#endif
}

#if AVBD_ENABLE_VIEWER_BRIDGE && defined(_WIN32)
void ViewerBridge::updateSnapshotStats(uint64_t bytes, float serializeMs, float sendMs, int sentClients, int failures)
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    metrics.snapshotCount++;
    metrics.lastSnapshotBytes = bytes;
    metrics.totalSnapshotBytes += bytes;
    metrics.lastSerializeMs = serializeMs;
    metrics.lastSendMs = sendMs;
    metrics.lastSentClients = sentClients;
    metrics.sendFailures += failures;
    metrics.clients = clientCount;
    metrics.avgSerializeMs += (serializeMs - metrics.avgSerializeMs) / (float)metrics.snapshotCount;
    metrics.avgSendMs += (sendMs - metrics.avgSendMs) / (float)metrics.snapshotCount;
}

void ViewerBridge::updateStatusStats()
{
    std::lock_guard<std::mutex> lock(metricsMutex);
    metrics.statusCount++;
    metrics.clients = clientCount;
}

void ViewerBridge::setStatusText(const char *message)
{
    std::lock_guard<std::mutex> lock(statusMutex);
    strcpy_s(status, message ? message : "");
}

void ViewerBridge::serverLoop(uint16_t port)
{
    WSADATA data = {};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
    {
        running = false;
        setStatusText("Viewer bridge failed: WSAStartup");
        return;
    }
    winsockStarted = true;

    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    listenSocket = (uintptr_t)listener;
    if (listener == INVALID_SOCKET)
    {
        running = false;
        setStatusText("Viewer bridge failed: socket");
        return;
    }

    BOOL reuse = TRUE;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse, sizeof(reuse));

    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (bind(listener, (sockaddr *)&address, sizeof(address)) == SOCKET_ERROR)
    {
        closesocket(listener);
        listenSocket = (uintptr_t)INVALID_SOCKET;
        running = false;
        setStatusText("Viewer bridge failed: bind 127.0.0.1:8765");
        return;
    }

    if (listen(listener, 4) == SOCKET_ERROR)
    {
        closesocket(listener);
        listenSocket = (uintptr_t)INVALID_SOCKET;
        running = false;
        setStatusText("Viewer bridge failed: listen");
        return;
    }

    u_long nonBlocking = 1;
    ioctlsocket(listener, FIONBIO, &nonBlocking);

    setStatusText("Viewer bridge listening on ws://127.0.0.1:8765");
    while (running)
    {
        SOCKET client = accept(listener, nullptr, nullptr);
        if (client != INVALID_SOCKET)
        {
            DWORD timeoutMs = 10;
            setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeoutMs, sizeof(timeoutMs));
            if (!performHandshake(client))
            {
                closesocket(client);
            }
            else
            {
                u_long clientNonBlocking = 1;
                ioctlsocket(client, FIONBIO, &clientNonBlocking);

                {
                    std::lock_guard<std::mutex> lock(clientsMutex);
                    clients.push_back(ViewerBridgeClient{(uintptr_t)client, false});
                    clientCount = (int)clients.size();
                }
                setStatusText("Viewer bridge connected");
            }
        }

        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            for (size_t i = 0; i < clients.size();)
            {
                SOCKET activeClient = asSocket(clients[i].socket);
                std::string payload;
                bool closed = false;
                bool receivedFrame = receiveTextFrame(activeClient, payload, closed);
                if (closed)
                {
                    closesocket(activeClient);
                    clients.erase(clients.begin() + i);
                    continue;
                }
                if (receivedFrame)
                {
                    SimulationCommand command;
                    if (parseCommandJson(payload, command))
                    {
                        bool binaryMode = clients[i].binarySnapshots;
                        if (handleBridgeLocalCommand(command, binaryMode))
                        {
                            clients[i].binarySnapshots = binaryMode;
                        }
                        else
                        {
                            std::lock_guard<std::mutex> commandLock(commandsMutex);
                            commands.push_back(command);
                        }
                    }
                }
                ++i;
            }
            clientCount = (int)clients.size();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
#endif
