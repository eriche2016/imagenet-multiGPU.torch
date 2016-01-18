--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then  -- 开的线程个数， 用于并行load data
      local options = opt -- make an upvalue to serialize over to donkey threads
      -- donkeys是管理者一组线程（在线程队列中）的线程池， opt.nDonkeys默认为2
      donkeys = Threads(  
         opt.nDonkeys,      -- 假设该脚本本身是主线程， 那么该函数或spawn nDonkeys个辅助线程， 或者子线程， 推入到thread队列中， 执行下面的f1, f2, ..的function list
         function()          
            require 'torch'
         end,
         function(idx)  -- idx是相关线程的线程号
            opt = options -- pass to all donkeys via upvalue
            tid = idx 
            local seed = opt.manualSeed + idx  -- 每个线程有自己的随机种子
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')  -- 所有子线程都会执行donkey.lua脚本
         end
      );
   else -- single threaded data loading. useful for debugging， 单线程情况
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

nClasses = nil
classes = nil
-- 训练数据工作
-- 一般而言， addjob（callback, endcallback, args), 中的callback（args）会由整个threads queue 并行化， 以便并行处理，
--  endcallback 是由main thread调用执行的。  注意前一个函数是有子线程执行， 返回的值会作为后一个函数（endcallback）的参数（是main thread执行的）
donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end) -- addjob函数会将一些函数插入到job queue中， 以便给所有的线程执行

-- 对线程池中的线程队列等进行一次同步操作， 确保上述所有数据加载工作完全结束
donkeys:synchronize()

nClasses = #classes
assert(nClasses, "Failed to get nClasses")
assert(nClasses == opt.nClasses,
       "nClasses is reported different in the data loader, and in the commandline options")
print('nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

-- 测试数据工作
nTest = 0
donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
donkeys:synchronize()
assert(nTest > 0, "Failed to get nTest")
print('nTest: ', nTest)
