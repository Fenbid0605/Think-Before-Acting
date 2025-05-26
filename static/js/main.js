// 全局变量
let sessionId = null;
let imageData = null;
let currentToolCalls = [];

// DOM元素
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const uploadImageBtn = document.getElementById('upload-image-btn');
const imageUpload = document.getElementById('image-upload');
const imagePreviewContainer = document.getElementById('image-preview-container');
const toolCallsContainer = document.getElementById('tool-calls-container');
const executeToolBtn = document.getElementById('execute-tool-btn');

// 工具参数定义
const toolParameters = {
    "realsense": {
        "initialize_camera": [],
        "stop_camera": [],
        "get_camera_frames": [],
        "get_point_cloud": [
            {name: "sample_rate", type: "number", default: 10, label: "采样率"}
        ],
        "get_depth_data": [],
        "get_object_distance": [
            {name: "x", type: "number", required: true, label: "X坐标"},
            {name: "y", type: "number", required: true, label: "Y坐标"},
            {name: "radius", type: "number", default: 5, label: "半径"}
        ]
    },
    "diana_robot": {
        "get_joint_positions": [],
        "get_tcp_position": [],
        "get_robot_state": [],
        "move_joints": [
            {name: "positions", type: "array", required: true, label: "关节位置数组 [7个值]"},
            {name: "velocity", type: "number", required: true, label: "速度比例 (0-1)"},
            {name: "acceleration", type: "number", required: true, label: "加速度比例 (0-1)"}
        ],
        "move_linear": [
            {name: "position", type: "array", required: true, label: "TCP位置 [x, y, z, rx, ry, rz]"},
            {name: "velocity", type: "number", required: true, label: "速度 (mm/s)"},
            {name: "acceleration", type: "number", required: true, label: "加速度 (mm/s²)"}
        ],
        "move_tcp_direction": [
            {name: "direction", type: "select", required: true, label: "方向", 
             options: [
                {value: 0, label: "X+"},
                {value: 1, label: "X-"},
                {value: 2, label: "Y+"},
                {value: 3, label: "Y-"},
                {value: 4, label: "Z+"},
                {value: 5, label: "Z-"}
             ]
            },
            {name: "velocity", type: "number", required: true, label: "速度 (mm/s)"},
            {name: "acceleration", type: "number", required: true, label: "加速度 (mm/s²)"}
        ],
        "rotate_tcp_direction": [
            {name: "direction", type: "select", required: true, label: "方向",
             options: [
                {value: 0, label: "Rx+"},
                {value: 1, label: "Rx-"},
                {value: 2, label: "Ry+"},
                {value: 3, label: "Ry-"},
                {value: 4, label: "Rz+"},
                {value: 5, label: "Rz-"}
             ]
            },
            {name: "velocity", type: "number", required: true, label: "角速度 (rad/s)"},
            {name: "acceleration", type: "number", required: true, label: "角加速度 (rad/s²)"}
        ],
        "stop_robot": [],
        "resume_robot": [],
        "enable_freedriving": [
            {name: "mode", type: "select", required: true, label: "模式",
             options: [
                {value: 0, label: "禁用"},
                {value: 1, label: "普通"},
                {value: 2, label: "强制"}
             ]
            }
        ],
        "release_brake": [],
        "hold_brake": []
    }
};

// 配置状态
let configStatus = {
    apiKeySet: false,
    model: "microsoft/phi-4-multimodal-instruct",
    temperature: 0.7,
    maxTokens: 4096
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检查API配置
    checkApiConfig();
    
    // 生成新的会话ID
    sessionId = generateUUID();
    
    // 添加事件监听器
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    uploadImageBtn.addEventListener('click', () => imageUpload.click());
    imageUpload.addEventListener('change', handleImageUpload);
    
    // 添加工具执行按钮事件监听器 - 连接到模态框的确认按钮
    document.getElementById('confirm-tool-btn').addEventListener('click', executeToolCall);
    
    // 添加欢迎消息
    addMessage('ai', [{ type: 'text', text: '你好！我是基于 Microsoft Phi-4 的多模态 AI 助手，由 Nvidia NIM 提供大模型计算。我可以查看图像并与机器人和深度相机交互。你可以直接向我提问。' }]);
});

// 检查API配置
async function checkApiConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        configStatus = {
            apiKeySet: config.api_key_set,
            model: config.model,
            temperature: config.temperature,
            maxTokens: config.max_tokens
        };
        
        if (!configStatus.apiKeySet) {
            // 提示用户设置API密钥
            addMessage('ai', [{ 
                type: 'text', 
                text: '⚠️ NVIDIA API密钥未设置。请在config.ini文件中配置API密钥后重启服务器。' 
            }]);
        }
    } catch (error) {
        console.error('获取配置失败:', error);
    }
}

// 发送消息
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message && !imageData) {
        return;
    }
    
    // 检查API密钥是否设置
    if (!configStatus.apiKeySet) {
        addMessage('ai', [{ 
            type: 'text', 
            text: '⚠️ 无法发送消息：NVIDIA API密钥未设置。请在config.ini文件中配置API密钥后重启服务器。' 
        }]);
        return;
    }
    
    // 显示用户消息
    const userContent = [];
    
    if (message) {
        userContent.push({ type: 'text', text: message });
    }
    
    if (imageData) {
        userContent.push({ type: 'image_url', image_url: { url: imageData } });
    }
    
    addMessage('user', userContent);
    
    // 清空输入
    userInput.value = '';
    clearImagePreview();
    
    // 显示加载中
    const loadingId = addLoadingMessage();
    
    try {
        // 发送请求
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: message,
                image_data: imageData
            })
        });
        
        const data = await response.json();
        
        // 移除加载消息
        removeMessage(loadingId);
        
        if (!data.success) {
            addMessage('ai', [{ type: 'text', text: `错误：${data.error}` }]);
            return;
        }
        
        // 更新会话ID
        sessionId = data.session_id;
        
        // 添加AI响应
        const assistantMessage = data.message;
        addMessage('ai', assistantMessage.content);
        
        // 处理工具调用
        if (data.tool_calls && data.tool_calls.length > 0) {
            handleToolCalls(data.tool_calls);
        }
    } catch (error) {
        console.error('发送消息失败:', error);
        removeMessage(loadingId);
        addMessage('ai', [{ type: 'text', text: `发送消息失败: ${error.message}` }]);
    }
}

// 添加消息到聊天界面
function addMessage(role, content) {
    const messageId = generateUUID();
    const messageElement = document.createElement('div');
    messageElement.id = `message-${messageId}`;
    messageElement.className = `message message-${role}`;
    
    let headerText = '';
    switch (role) {
        case 'user':
            headerText = '用户';
            break;
        case 'ai':
            headerText = 'AI助手';
            break;
        case 'function':
            headerText = '工具调用结果';
            break;
    }
    
    let messageHTML = `
        <div class="message-header">${headerText}</div>
        <div class="message-content">
    `;
    
    // 处理内容
    if (Array.isArray(content)) {
        for (const item of content) {
            if (item.type === 'text') {
                // 使用marked处理Markdown
                messageHTML += marked.parse(item.text);
            } else if (item.type === 'image_url') {
                messageHTML += `<img src="${item.image_url.url}" alt="图像" />`;
            }
        }
    } else if (typeof content === 'string') {
        // 简单文本消息
        messageHTML += content;
    }
    
    messageHTML += '</div>';
    messageElement.innerHTML = messageHTML;
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// 添加加载中消息
function addLoadingMessage() {
    const messageId = generateUUID();
    const messageElement = document.createElement('div');
    messageElement.id = `message-${messageId}`;
    messageElement.className = 'message message-ai';
    
    messageElement.innerHTML = `
        <div class="message-header">AI助手</div>
        <div class="message-content">
            <div class="spinner-border spinner-border-sm text-secondary" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <span class="ms-2">思考中...</span>
        </div>
    `;
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// 移除消息
function removeMessage(messageId) {
    const messageElement = document.getElementById(`message-${messageId}`);
    if (messageElement) {
        messageElement.remove();
    }
}

// 处理工具调用
function handleToolCalls(toolCalls) {
    currentToolCalls = toolCalls;
    
    // 清空工具调用容器
    toolCallsContainer.innerHTML = '';
    
    // 为每个工具调用添加卡片
    for (let i = 0; i < toolCalls.length; i++) {
        const tool = toolCalls[i];
        const toolId = `tool-${generateUUID()}`;
        const toolCard = document.createElement('div');
        toolCard.className = 'tool-card';
        toolCard.id = toolId;
        
        const categoryLabel = tool.category === 'realsense' ? '深度相机' : '机器人';
        
        toolCard.innerHTML = `
            <div class="tool-card-header">
                <div>
                    <h6 class="tool-name">${tool.tool}</h6>
                    <div class="tool-category">${categoryLabel}</div>
                </div>
                <span class="tool-status tool-status-pending">待执行</span>
            </div>
            <div class="tool-card-body">
                <button class="btn btn-sm btn-primary execute-tool-btn" data-index="${i}">
                    设置参数并执行
                </button>
            </div>
        `;
        
        toolCallsContainer.appendChild(toolCard);
        
        // 添加执行按钮点击事件
        toolCard.querySelector('.execute-tool-btn').addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.index);
            showToolParamsModal(index);
        });
    }
}

// 显示工具参数设置弹窗
function showToolParamsModal(toolIndex) {
    const tool = currentToolCalls[toolIndex];
    const toolParamsModal = new bootstrap.Modal(document.getElementById('toolParamsModal'));
    const paramsContainer = document.getElementById('params-container');
    const modalTitle = document.getElementById('toolParamsModalLabel');
    
    // 设置标题
    modalTitle.textContent = `${tool.tool} 参数设置`;
    
    // 清空参数容器
    paramsContainer.innerHTML = '';
    
    // 获取参数定义
    const paramDefs = toolParameters[tool.category][tool.tool] || [];
    
    // 清空参数容器，但保留隐藏字段
    paramsContainer.innerHTML = '';
    
    // 添加工具索引的隐藏字段
    const indexInput = document.createElement('input');
    indexInput.type = 'hidden';
    indexInput.id = 'tool-index';
    indexInput.value = toolIndex;
    paramsContainer.appendChild(indexInput);
    
    if (paramDefs.length === 0) {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'alert alert-info';
        infoDiv.textContent = '此工具无需参数';
        paramsContainer.appendChild(infoDiv);
    } else {
        // 为每个参数创建输入控件
        for (const param of paramDefs) {
            const formGroup = document.createElement('div');
            formGroup.className = 'mb-3';
            
            const label = document.createElement('label');
            label.className = 'form-label';
            label.htmlFor = `param-${param.name}`;
            label.textContent = `${param.label}${param.required ? ' *' : ''}`;
            
            let input;
            
            if (param.type === 'select') {
                input = document.createElement('select');
                input.className = 'form-select';
                
                for (const option of param.options) {
                    const optionEl = document.createElement('option');
                    optionEl.value = option.value;
                    optionEl.textContent = option.label;
                    input.appendChild(optionEl);
                }
            } else if (param.type === 'array') {
                input = document.createElement('input');
                input.type = 'text';
                input.className = 'form-control';
                input.placeholder = '使用逗号分隔的数值，例如: 1.0, 2.0, 3.0';
            } else {
                input = document.createElement('input');
                input.type = param.type === 'number' ? 'number' : 'text';
                input.className = 'form-control';
            }
            
            input.id = `param-${param.name}`;
            input.name = param.name;
            
            if (param.default !== undefined) {
                input.value = param.default;
            }
            
            if (param.required) {
                input.required = true;
            }
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
            
            paramsContainer.appendChild(formGroup);
        }
    }
    
    toolParamsModal.show();
}

// 执行工具调用
async function executeToolCall() {
    // 检查工具索引元素是否存在
    const toolIndexElement = document.getElementById('tool-index');
    if (!toolIndexElement) {
        console.error('工具索引元素不存在');
        return;
    }
    
    const toolIndex = parseInt(toolIndexElement.value);
    
    // 检查索引是否有效
    if (isNaN(toolIndex) || toolIndex < 0 || toolIndex >= currentToolCalls.length) {
        console.error('无效的工具索引');
        return;
    }
    
    const tool = currentToolCalls[toolIndex];
    const toolCards = document.querySelectorAll('.tool-card');
    
    // 检查工具卡片是否存在
    if (!toolCards || toolIndex >= toolCards.length) {
        console.error('工具卡片不存在');
        return;
    }
    
    const toolId = toolCards[toolIndex].id;
    
    // 获取参数
    const parameters = {};
    const paramDefs = toolParameters[tool.category][tool.tool] || [];
    
    for (const param of paramDefs) {
        const input = document.getElementById(`param-${param.name}`);
        
        if (input) {
            let value = input.value;
            
            // 处理数组
            if (param.type === 'array') {
                value = value.split(',').map(v => {
                    const num = parseFloat(v.trim());
                    return isNaN(num) ? v.trim() : num;
                });
            }
            // 处理数字
            else if (param.type === 'number') {
                value = parseFloat(value);
            }
            // 处理选择
            else if (param.type === 'select') {
                value = parseInt(value);
            }
            
            parameters[param.name] = value;
        }
    }
    
    // 关闭模态框
    const modal = bootstrap.Modal.getInstance(document.getElementById('toolParamsModal'));
    modal.hide();
    
    // 更新工具卡片状态
    const toolCard = document.getElementById(toolId);
    const statusSpan = toolCard.querySelector('.tool-status');
    const toolBody = toolCard.querySelector('.tool-card-body');
    
    statusSpan.className = 'tool-status tool-status-pending';
    statusSpan.textContent = '执行中';
    toolBody.innerHTML = `
        <div class="d-flex justify-content-center py-3">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">执行中...</span>
            </div>
        </div>
    `;
    
    try {
        // 调用工具
        const response = await fetch('/api/tool-call', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                tool: `${tool.category}/${tool.tool}`,
                parameters: parameters
            })
        });
        
        const data = await response.json();
        
        // 更新工具卡片
        if (data.success) {
            statusSpan.className = 'tool-status tool-status-completed';
            statusSpan.textContent = '已完成';
            
            toolBody.innerHTML = `
                <div class="tool-result">
                    ${JSON.stringify(data.result, null, 2)}
                </div>
            `;
            
            // 添加函数调用消息到聊天
            const functionName = `${tool.category}/${tool.tool}`;
            addMessage('function', `${functionName} 调用结果:\n${JSON.stringify(data.result, null, 2)}`);
            
            // 如果是获取图像，显示图像
            if (tool.tool === 'get_camera_frames' && data.result.success) {
                const imageMessage = document.createElement('div');
                imageMessage.className = 'message message-ai';
                
                imageMessage.innerHTML = `
                    <div class="message-header">深度相机图像</div>
                    <div class="message-content">
                        <div class="row">
                            <div class="col-6">
                                <p>彩色图像:</p>
                                <img src="data:image/jpeg;base64,${data.result.color_image}" alt="彩色图像" />
                            </div>
                            <div class="col-6">
                                <p>深度图像:</p>
                                <img src="data:image/jpeg;base64,${data.result.depth_image}" alt="深度图像" />
                            </div>
                        </div>
                    </div>
                `;
                
                chatMessages.appendChild(imageMessage);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        } else {
            statusSpan.className = 'tool-status tool-status-error';
            statusSpan.textContent = '失败';
            
            toolBody.innerHTML = `
                <div class="alert alert-danger">
                    ${data.error || '工具调用失败'}
                </div>
            `;
            
            // 添加错误消息到聊天
            addMessage('function', `工具调用失败: ${data.error}`);
        }
    } catch (error) {
        console.error('工具调用失败:', error);
        
        statusSpan.className = 'tool-status tool-status-error';
        statusSpan.textContent = '错误';
        
        toolBody.innerHTML = `
            <div class="alert alert-danger">
                ${error.message}
            </div>
        `;
        
        // 添加错误消息到聊天
        addMessage('function', `工具调用错误: ${error.message}`);
    }
}

// 处理图片上传
function handleImageUpload(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
        imageData = e.target.result;
        
        // 显示图片预览
        imagePreviewContainer.innerHTML = `
            <div class="image-preview">
                <img src="${imageData}" alt="上传的图片" />
                <span class="remove-image" title="移除图片">×</span>
            </div>
        `;
        
        // 添加移除图片事件
        const removeBtn = imagePreviewContainer.querySelector('.remove-image');
        removeBtn.addEventListener('click', clearImagePreview);
    };
    
    reader.readAsDataURL(file);
}

// 清除图片预览
function clearImagePreview() {
    imagePreviewContainer.innerHTML = '';
    imageData = null;
    imageUpload.value = '';
}

// 生成UUID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
} 